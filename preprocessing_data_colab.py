from preprocessor import Preprocessor

prep = Preprocessor()
prep.preprocess_data(annot_path="train_annotations.csv",new_annot_path="processed_train_annotations.csv", resize_dims=(32, 32), new_images_path = "processed_train")
prep.preprocess_data(annot_path="test_annotations.csv",new_annot_path="processed_test_annotations.csv", resize_dims=(32, 32), new_images_path = "processed_test")

prep.save_data_to_npfile("processed_test", "processed_test_annotations.csv", "test_imgs.npy", "test_labels.npy")
prep.save_data_to_npfile("processed_test", "processed_test_annotations.csv", "test_imgs.npy", "test_labels.npy")

# imgs = prep.load_from_array("test_imgs.npy")
# lbs = prep.load_from_array("test_labels.npy")
