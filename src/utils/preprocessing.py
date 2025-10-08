import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def map_to_8_classes(label_mask, mapping):
    new_mask = np.ones_like(label_mask, dtype=np.uint8) * 255  # classe ignorée
    for label_id, class_8 in mapping.items():
        new_mask[label_mask == label_id] = class_8
    return new_mask

def resize_and_save(image_path, mask_path, output_dir_img, output_dir_mask, size=(256, 512)):
    """
    Redimensionne une image et son masque associé, puis les sauvegarde dans des dossiers distincts.
    """
    # Charger l'image (RGB)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Charger le masque (uint8 brut labelId)
    label_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Convertir le masque en 8 classes
    converted_mask = map_to_8_classes(label_mask, LABELID_TO_8CLASS)

    # Resize
    image_resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(converted_mask, size, interpolation=cv2.INTER_NEAREST)

    # Préparer les noms
    name = os.path.basename(image_path).replace("_leftImg8bit.png", "")
    img_out_path = os.path.join(output_dir_img, f"{name}.png")
    mask_out_path = os.path.join(output_dir_mask, f"{name}_mask.png")

    # Créer les dossiers si besoin
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)
    
    # Enregistrer sur disque
    # Attention : OpenCV attend BGR pour l'image
    cv2.imwrite(img_out_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_out_path, mask_resized)

def process_all_splits(
    root_images="../data/raw/leftImg8bit",
    root_masks="../data/raw/gtFine",
    out_img_root="../data/processed/images_256x512",
    out_mask_root="../data/processed/masks_256x512",
    size=(256, 512)
):
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"\n Traitement split : {split}")

        split_img_dir = os.path.join(root_images, split)
        split_mask_dir = os.path.join(root_masks, split)

        for city in os.listdir(split_img_dir):
            img_city_dir = os.path.join(split_img_dir, city)
            mask_city_dir = os.path.join(split_mask_dir, city)

            for filename in os.listdir(img_city_dir):
                if filename.endswith("_leftImg8bit.png"):
                    name_base = filename.replace("_leftImg8bit.png", "")
                    img_path = os.path.join(img_city_dir, filename)
                    mask_path = os.path.join(mask_city_dir, f"{name_base}_gtFine_labelIds.png")

                    # Sortie
                    out_img_dir = os.path.join(out_img_root, split)
                    out_mask_dir = os.path.join(out_mask_root, split)

                    try:
                        resize_and_save(
                            img_path, mask_path,
                            out_img_dir, out_mask_dir,
                            size
                        )
                    except Exception as e:
                        print(f" Erreur avec {img_path} : {e}")

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(image, mask):
    feature = {
        "image": _bytes_feature(image),
        "mask": _bytes_feature(mask)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def convert_to_tfrecords(img_dir, mask_dir, output_path):
    image_filenames = sorted(os.listdir(img_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    assert len(image_filenames) == len(mask_filenames), "Image et masque : tailles différentes !"

    with tf.io.TFRecordWriter(output_path) as writer:
        for img_name, mask_name in tqdm(zip(image_filenames, mask_filenames), total=len(image_filenames)):
            # Lecture
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # attention : mask = uint8 avec classes

            # Vérification
            if image is None or mask is None:
                print(f"⚠️ Fichier manquant ou corrompu : {img_name}")
                continue

            # Conversion en bytes
            image_bytes = tf.io.encode_png(image).numpy()
            
            # Ajouter une dimension au masque
            mask = np.expand_dims(mask, axis=-1)  # (H, W) → (H, W, 1)
            mask_bytes = tf.io.encode_png(mask).numpy()

            example = serialize_example(image_bytes, mask_bytes)
            writer.write(example)

    print(f"TFRecord créé ({output_path}) avec {len(image_filenames)} exemples.")

