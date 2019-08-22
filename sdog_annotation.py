import xml.etree.ElementTree
import sys
import os

import tqdm

# classes = ["scottish_deerhound", "silky_terrier"]
split = 100
all_classes = [
    'Chihuahua'
    , 'Japanese_spaniel'
    , 'Maltese_dog'
    , 'Pekinese'
    , 'Shih-Tzu'
    , 'Blenheim_spaniel'
    , 'papillon'
    , 'toy_terrier'
    , 'Rhodesian_ridgeback'
    , 'Afghan_hound'
    , 'basset'
    , 'beagle'
    , 'bloodhound'
    , 'bluetick'
    , 'black-and-tan_coonhound'
    , 'Walker_hound'
    , 'English_foxhound'
    , 'redbone'
    , 'borzoi'
    , 'Irish_wolfhound'
    , 'Italian_greyhound'
    , 'whippet'
    , 'Ibizan_hound'
    , 'Norwegian_elkhound'
    , 'otterhound'
    , 'Saluki'
    , 'Scottish_deerhound'
    , 'Weimaraner'
    , 'Staffordshire_bullterrier'
    , 'American_Staffordshire_terrier'
    , 'Bedlington_terrier'
    , 'Border_terrier'
    , 'Kerry_blue_terrier'
    , 'Irish_terrier'
    , 'Norfolk_terrier'
    , 'Norwich_terrier'
    , 'Yorkshire_terrier'
    , 'wire-haired_fox_terrier'
    , 'Lakeland_terrier'
    , 'Sealyham_terrier'
    , 'Airedale'
    , 'cairn'
    , 'Australian_terrier'
    , 'Dandie_Dinmont'
    , 'Boston_bull'
    , 'miniature_schnauzer'
    , 'giant_schnauzer'
    , 'standard_schnauzer'
    , 'Scotch_terrier'
    , 'Tibetan_terrier'
    , 'silky_terrier'
    , 'soft-coated_wheaten_terrier'
    , 'West_Highland_white_terrier'
    , 'Lhasa'
    , 'flat-coated_retriever'
    , 'curly-coated_retriever'
    , 'golden_retriever'
    , 'Labrador_retriever'
    , 'Chesapeake_Bay_retriever'
    , 'German_short-haired_pointer'
    , 'vizsla'
    , 'English_setter'
    , 'Irish_setter'
    , 'Gordon_setter'
    , 'Brittany_spaniel'
    , 'clumber'
    , 'English_springer'
    , 'Welsh_springer_spaniel'
    , 'cocker_spaniel'
    , 'Sussex_spaniel'
    , 'Irish_water_spaniel'
    , 'kuvasz'
    , 'schipperke'
    , 'groenendael'
    , 'malinois'
    , 'briard'
    , 'kelpie'
    , 'komondor'
    , 'Old_English_sheepdog'
    , 'Shetland_sheepdog'
    , 'collie'
    , 'Border_collie'
    , 'Bouvier_des_Flandres'
    , 'Rottweiler'
    , 'German_shepherd'
    , 'Doberman'
    , 'miniature_pinscher'
    , 'Greater_Swiss_Mountain_dog'
    , 'Bernese_mountain_dog'
    , 'Appenzeller'
    , 'EntleBucher'
    , 'boxer'
    , 'bull_mastiff'
    , 'Tibetan_mastiff'
    , 'French_bulldog'
    , 'Great_Dane'
    , 'Saint_Bernard'
    , 'Eskimo_dog'
    , 'malamute'
    , 'Siberian_husky'
    , 'affenpinscher'
    , 'basenji'
    , 'pug'
    , 'Leonberg'
    , 'Newfoundland'
    , 'Great_Pyrenees'
    , 'Samoyed'
    , 'Pomeranian'
    , 'chow'
    , 'keeshond'
    , 'Brabancon_griffon'
    , 'Pembroke'
    , 'Cardigan'
    , 'toy_poodle'
    , 'miniature_poodle'
    , 'standard_poodle'
    , 'Mexican_hairless'
    , 'dingo'
    , 'dhole'
    , 'African_hunting_dog'
]
train_classes = all_classes[:split]
test_classes = all_classes[split:]
class2id = dict(zip(train_classes, range(len(train_classes))))


def parse_annotation(path):
    xml_root = xml.etree.ElementTree.parse(path).getroot()
    object = xml_root.findall('object')[0]
    name = object.findall('name')[0].text
    bound_box = object.findall('bndbox')[0]
    fn = os.path.join(xml_root.findall('folder')[0].text.lower() + "-" + name,
                      xml_root.findall('filename')[0].text.lower() + ".jpg")
    return (
        path.replace("Annotation", "Images") + '.jpg'
        , bound_box.findall('xmin')[0].text
        , bound_box.findall('ymin')[0].text
        , bound_box.findall('xmax')[0].text
        , bound_box.findall('ymax')[0].text
        , class2id.get(name.lower())
    )


def main(argv):
    input, output_fn = argv

    images_root_dir = os.path.join(input, 'Images')
    annotations_root_dir = os.path.join(input, 'Annotation')
    with open(output_fn, "w") as output_file:
        for breed_dir in tqdm.tqdm([d for d in os.listdir(annotations_root_dir)]):
            # print(breed_dir)
            for annotation_file in [f for f in os.listdir(os.path.join(annotations_root_dir, breed_dir))]:
                annotation = parse_annotation(os.path.join(annotations_root_dir, breed_dir, annotation_file))
                if annotation[-1] is not None:
                    output_file.write("{} {},{},{},{},{}\n".format(*annotation))


if __name__ == '__main__':
    main(sys.argv[1:])
