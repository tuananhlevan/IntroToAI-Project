import splitfolders

input_folder = "IntroToAI-Project\data\KidneyCancer"
output_folder = "IntroToAI-Project\data_split\KidneyCancer"

splitfolders.ratio(input=input_folder,
                   output=output_folder,
                   seed=42,
                   ratio=(.8, .1, .1))