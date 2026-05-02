import h5py
import numpy as np
import cv2
import os
import argparse
import sys
from tqdm import tqdm
import multiprocessing
from glob import glob


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out+'\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.6f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'int':
            self._write('{}: {}'.format(key, value))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'int':
            output = int(self.fs.getNode(key).real())
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract data from SMC files using multiprocessing.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the directory containing main .smc files.")
    parser.add_argument('--output_root_dir', type=str, required=True, help="Path to the root directory for extracted data.")
    parser.add_argument('--num_processes', type=int, default=8, help="Number of processes to use for parallel extraction. Default is 8.")
    return parser.parse_args()

def write_camera(camera, path, suffix=None):
    from os.path import join
    intri_name = join(path, 'intri.yml') if suffix is None else join(path, 'intri_{}.yml'.format(suffix))
    extri_name = join(path, 'extri.yml') if suffix is None else join(path, 'extri_{}.yml'.format(suffix))
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['dist'])
        if 'H' in val.keys() and 'W' in val.keys():
            intri.write('H_{}'.format(key), val['H'], dt='int')
            intri.write('W_{}'.format(key), val['W'], dt='int')
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])

def process_and_save_image(args):
    frame_id, img_buffer, output_file_path = args
    try:
        img = cv2.imdecode(np.frombuffer(img_buffer, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return (frame_id, "Decode failed")
        cv2.imwrite(output_file_path, img)
        return (frame_id, "Success")
    except Exception as e:
        return (frame_id, str(e))

def main(args):
    input_path = args.input_dir
    part_idx = input_path.rstrip('/').split('/')[-1].split('_')[-2][-1]
    output_root_dir = args.output_root_dir
    num_processes = args.num_processes
    main_file_list = glob(os.path.join(input_path, '*.smc'))
    
    main_file_list.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    for main_file in main_file_list: 
        seq_id = main_file.split('/')[-1][:-4]
        print(f"Processing: {seq_id}")
                
        output_dir = os.path.join(output_root_dir, seq_id)
        os.makedirs(output_dir, exist_ok=True)

        # --- 1. Extract RGB camera parameters (main.smc) --- 
        print(f"Extracting RGB camera parameters...")
        rgb_intri_path = os.path.join(output_dir, 'intri.yml')
        rgb_extri_path = os.path.join(output_dir, 'extri.yml')
        if os.path.exists(rgb_intri_path) and os.path.exists(rgb_extri_path):
            print(f"File '{rgb_intri_path}' and '{rgb_extri_path}' already exists. Skipping RGB camera extraction.")
        else: 
            if int(part_idx) < 3: # Part 1~2
                annots_file = os.path.join(input_path, '..', f'dna_rendering_part{part_idx}_annotations', f'{seq_id}_annots.smc')
                if not os.path.exists(annots_file):
                    print(f"Error: Annotations data file not found at {annots_file}")
                    continue
                file = annots_file
            else:  # Part 3~6
                file = main_file
            try: 
                rgb_cameras = {}
                with h5py.File(file, 'r') as f:
                    if 'Camera_Parameter' in f:
                        param_group = f['Camera_Parameter']
                        for cam_id, cam_data in tqdm(param_group.items(), desc='Parsing RGB cameras'):
                            K = np.asarray(cam_data['K'][()])
                            dist = np.asarray([cam_data['D'][()]])
                            RT_c2w = np.asarray(cam_data['RT'][()])
                            RT_w2c = np.linalg.inv(RT_c2w)   # c2w to w2c
                            
                            rgb_cameras[cam_id] = {
                                'K': K, 
                                'dist': dist, 
                                'R': RT_w2c[:3, :3], 
                                'T': RT_w2c[:3, 3:]
                            }
                        
                        keys = sorted(list(rgb_cameras.keys()))
                        rgb_cameras = {key: rgb_cameras[key] for key in keys}
                        
                        write_camera(rgb_cameras, output_dir)
                        print(f"RGB cameras saved to {rgb_intri_path} and {rgb_extri_path}")
                    else:
                        print(f"Warning: 'Camera_Parameter' not found in {file}")
            except Exception as e:
                print(f"Error reading RGB camera data: {e}")            
                
        # --- 2. Extract images (main.smc) ---
        img_dir = os.path.join(output_dir, 'images_orig')
        if os.path.exists(img_dir):
            print(f"Directory '{img_dir}' already exists. Skipping image extraction.")
        else:
            print("Preparing image extraction tasks...")
            tasks_image = []
            with h5py.File(main_file, 'r') as f:
                for cam_type in ['Camera_5mp', 'Camera_12mp']:
                    if cam_type in f:
                        camera_group = f[cam_type]
                        for camera_id in camera_group:
                            cam_img_dir = os.path.join(img_dir, f'{int(camera_id):02d}')
                            os.makedirs(cam_img_dir, exist_ok=True)
                            frames = camera_group[camera_id]['color']
                            for frame_id in frames:
                                output_file_path = os.path.join(cam_img_dir, f'{int(frame_id):06d}.jpg')
                                img_buffer = frames[frame_id][()].tobytes()
                                # Remove ccm
                                tasks_image.append((int(frame_id), img_buffer, output_file_path))

            if tasks_image:
                print(f"Total {len(tasks_image)} images to process. Starting parallel extraction...")
                print(f"Using {num_processes} processes for image extraction...")
                with multiprocessing.Pool(processes=num_processes) as pool:
                    list(tqdm(pool.imap_unordered(process_and_save_image, tasks_image), total=len(tasks_image), desc="Extracting images"))
                print("\nImage extraction finished.")
            else:
                print("No images found to extract.")
        print("------------------------------------------------")

    print("\nExtraction process finished successfully.")

if __name__ == '__main__':
    args = parse_args()
    main(args)