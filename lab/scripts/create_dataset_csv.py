import glob
import os

dataset_root = '/work/cvcs2026/latent_space_cowboys/datasets/HemoSet'

CSV_FILE_NAME = 'splits/dataset.csv'

os.makedirs('splits', exist_ok=True)

with open(CSV_FILE_NAME, 'w') as f:
    f.write('images,labels,video_id\n')

    for pig_number in range(1, 12):
        pig_name = f'pig{pig_number}'

        folder_dir_images = f'{dataset_root}/{pig_name}/imgs'
        folder_dir_labels = f'{dataset_root}/{pig_name}/labels'

        print(f'Leggo {pig_name}')

        for image_path in sorted(glob.iglob(f'{folder_dir_images}/*')):

            if not image_path.endswith('.png'):
                continue

            image_filename = os.path.basename(image_path)
            name_without_extension = os.path.splitext(image_filename)[0]

            label_filename = f'{name_without_extension}_mask.png'
            label_path = f'{folder_dir_labels}/{label_filename}'

            if not os.path.exists(label_path):
                print(f'Label mancante per: {image_path}')
                print(f'Cercavo: {label_path}')
                continue

            f.write(image_path)
            f.write(',')
            f.write(label_path)
            f.write(',')
            f.write(pig_name)
            f.write('\n')

print(f'Creato file CSV: {CSV_FILE_NAME}')