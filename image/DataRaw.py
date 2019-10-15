from functools import partial
from typing import *
from pathlib import *

import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

import numpy as np
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import seaborn as sns
import re

from .utils.tf_resizer import tf_resizer
from .utils.utils import *



class DataRaw:
    """
    This class creates a dataframe with image locations and labels, and stores the most important information
    about the labels and image files. Instances of this class can then be used with DataSplit. Images need to
    be stored in a 'train' folder.
    DataRaw methods can be used to do basic exploration of the raw data, such as label descriptions
    and distributions (desribe), displaying images with unusually skewed dimensions (show_skewed_images).

    # Example
        ```python
            # Images stored in '/path/to_images/train'
            data_dir = '/path/to_images'
            my_data = DataRaw(data_dir, filetype="png")

            # Initiate a DataFrame (can be used with existing csv or read filenames/labels automatically)
            test_data.init_df()

            # Encode labels type. Pass a dictionary with values: labels, or use automatic encoding.
            test_data.encode(categorical=True, label_map="auto", one_hot=True, multilabel=True)

            # Explore
            test_data.df.value_counts()
            test_data.show_images()
            test_data.describe()
        ```

    # Arguments
        path: path to the data directory that contains the 'train' data folder.
        filetype: filetype of the images (e.g. "png", "gif", "jpeg". No dot!)

    """

    def __init__(self, path, filetype):
        self.path = str(path)
        self.filetype = filetype

    def init_df(self,
        dataframe_path: Union[str, Path] = None,
        label_col: str = None,
        id_col: str = None,
        sep: Optional[str] = ",",
        y_source: Optional[str] = "filenames",
        y_regex: Optional[str] = "",
        get_resolution: Optional[bool] = False):

        """
        This method creates a dataframe DataRaw.df that contains filepaths and labels. The dataframe can
        be created from an existing csv file, or by scanning the DataRaw.path/train for image files. Labels
        can be parsed from folder names or filenames (using regex). If CSV dataframe is used, it will be
        tidied up to contain full paths to image files and filetypes.

        # Examples
            ```python
                    # Create a dataframe from an existing train.csv file in DataRaw.path directory:
                    my_data = DataRaw(...)
                    my_data.init_df(dataframe_path=my_data.path + "/train.csv")

                    # Create a dataframe based on images in DataRaw.path and parse filenames for labels:
                    my_data.init_df(dataframe_path=None, y_source="filenames", y_regex=".*_(.*).png")

                    # Create labels based on directory names:
                    my_data.init_df(dataframe_path=None, y_source="dirnames")
            ```

        # Arguments
            dataframe_path: path to the dataframe csv file that contains image locations and labels. If not
                provided, y_source argument needs to be provided.
            sep: separator of the dataframe csv provided in dataframe_path. Default - ","
            y_source: source of the image labels, if dataframe_path is None. "filenames" parses filenames with
                provided y_regex argument, "dirnames" parses directory names.
            y_regex: string that provides regex for label extraction from filenames, if y_source == "filenames".
                E.g.: for "file_location/image_file_label1.png", provided y_regex = ".*_(.*).png", would result in
                "label1" label.
            get_resolution: add "height" and "weight" dimensions for each image to the dataframe (takes some time).
        """

        if dataframe_path == None:
            self.df = self._df_no_csv(y_source, y_regex, get_resolution)

        if (dataframe_path != None):
            self.df = self._df_from_csv(dataframe_path, label_col, id_col, sep, get_resolution)

    def _df_from_csv(self,
        dataframe_path: str,
        label_col: str,
        id_col: str,
        sep: str,
        get_resolution: bool) -> pd.DataFrame:
        """
        Method is used if DataRaw.init_df(dataframe_path) is provided. Reads the provided csv into DataRaw.df,
        checks and appends image file extensions if necessary, prepends full paths to the file. Returns a dataframe to
        DataRaw.df

        # Arguments
            dataframe_path: path to the dataframe csv file that contains image locations and labels. If not
                provided, y_source argument needs to be provided.
            sep: separator of the dataframe csv provided in dataframe_path. Default - ","
            get_resolution: add "height" and "weight" dimensions for each image to the dataframe (takes some time).
        """
        df = pd.read_csv(dataframe_path, sep=sep)
        if (id_col != "id") & ("id" in df.columns):
            df = df.rename(mapper={"id": "id_from_csv"}, axis=1)
        if (label_col != "label") & ("label" in df.columns) in df.columns:
            df = df.rename(mapper={"label": "label_from_csv"}, axis=1)
        df = pd.concat([df[[id_col, label_col]], df.drop([id_col, label_col], axis=1)], axis=1)
        df_cols = list(df.columns)
        df_cols[:2] = ["id", "label"]
        df.columns = df_cols
        df = self._extension_check(df)
        df = self._prepend_parent_dirs(df)
        if get_resolution:
            df["height"] = df["id"].map(partial(utils.get_image_resolution, dim=0))
            df["width"] = df["id"].map(partial(utils.get_image_resolution, dim=1))
        df.reset_index(inplace=True, drop=True)
        return df


    def _df_no_csv(self,
        y_source: str,
        y_regex: str,
        get_resolution: bool) -> pd.DataFrame:
        """
        Method is used if DataRaw.init_df has no csv provided. Reads image filenames from self.path/train,
        parses labels from either "filenames" (using "y_regex") or "dirnames". Returns a dataframe to
        DataRaw.df.

        # Arguments
            y_source: source of the image labels, if dataframe_path is None. "filenames" parses filenames with
                provided y_regex argument, "dirnames" parses directory names.
            y_regex: string that provides regex for label extraction from filenames, if y_source == "filenames".
                E.g.: for "file_location/image_file_label1.png", provided y_regex = ".*_(.*).png", would result in
                "label1" label.
            get_resolution: add "height" and "weight" dimensions for each image to the dataframe (takes some time).
        """
        df = pd.DataFrame({"id": utils.file_getter(self.path+'/train')})
        if y_source == "filenames":
            if y_regex == "":
                raise ValueError("Please enter a y_regex value to mach the filenames.")
            df["label"] = df["id"].str.extract(y_regex)
        elif y_source == "dirnames":
            df["label"] = df["id"].map(lambda x: re.match('.*/(.*)/.*$', x).group(1))
        if get_resolution:
            df["height"] = df["id"].map(partial(utils.get_image_resolution, dim=0))
            df["width"] = df["id"].map(partial(utils.get_image_resolution, dim=1))
        df.reset_index(inplace=True, drop=True)
        return df


    def label_encode(self,
        categorical: bool=True,
        label_map: Optional[Union[str, dict]] = "auto",
        one_hot: bool = False,
        multilabel: bool = False) -> None:
        """
        Method encodes labels in DataRaw.df "label column" and declares label type. Used after DataRaw.init_df()
        instantiates a dataframe.

        # Arguments
            categorical: is the label categorical or continuous (classification or regression problem).
            label_map: use "auto" to automatically encode labels using data in the "label" column or, optionally,
                pass a dictionary with "value: label" items.
            one_hot: encode labels using one_hot encoding? If so, columns with labels as column names are created,
                keeping the original "label" column.
            multilabel: are labels multilabel? If true, each value of "label" column has to contain a List with
                values/labels.
        """

        self.y_categorical = categorical
        self.y_multilabel = multilabel

        if not multilabel:
            if label_map == "auto":
                if one_hot:
                    lb = LabelBinarizer()
                    lb.fit(self.df.label)
                    self.label_map = dict(enumerate(lb.classes_))
                    one_hot_cols = lb.transform(self.df.label)
                    self.df = pd.concat([self.df, pd.DataFrame(one_hot_cols, columns=lb.classes_)], axis=1)
                    self.df.columns = self.df.columns[:-len(self.label_map)].append(pd.Index(self.label_map.values()))
                else:
                    self.label_map = dict(enumerate(self.df.label.unique()))
                    reverse_dict = dict([(v,k) for k,v in self.label_map.items()])
                    self.df.label = self.df.label.map(reverse_dict)
            elif label_map != "auto":
                self.label_map = label_map
                if one_hot:
                    if type(self.df.label[0]) == str:
                        reverse_dict = dict([(v,k) for k,v in self.label_map.items()])
                        self.df.label = self.df.label.map(reverse_dict)
                    one_hot_cols = pd.get_dummies(self.df.label.map(self.label_map))
                    self.df = pd.concat([self.df, one_hot_cols], axis=1)
                elif not one_hot:
                    if (type(self.df.label[0]) == str):
                        reverse_dict = dict([(v,k) for k,v in self.label_map.items()])
                        self.df.label = self.df.label.map(reverse_dict)

        if multilabel:
            if one_hot == False:
                print("Multilabel outcome var requested. One_hot = True assumed.")
                one_hot = True
                self.y_one_hot = True

            if label_map == "auto":
                mlb = MultiLabelBinarizer()
                mlb.fit(self.df.label)
                self.label_map = dict(enumerate(mlb.classes_))
                one_hot_cols = mlb.transform(self.df.label)
                self.df = pd.concat([self.df, pd.DataFrame(one_hot_cols, columns=mlb.classes_)], axis=1)
                self.df.columns = self.df.columns[:-len(self.label_map)].append(pd.Index(self.label_map.values()))

            else:
                if type(self.df.label[0]) == int:
                    mlb = MultiLabelBinarizer(classes=list(label_map.keys()))
                elif type(self.df.label[0]) == str:
                    mlb = MultiLabelBinarizer(classes=list(label_map.values()))
                mlb.fit(self.df.label)
                self.label_map = label_map
                one_hot_cols = mlb.transform(self.df.label)
                self.df = pd.concat([self.df, pd.DataFrame(one_hot_cols, columns=mlb.classes_)], axis=1)
                self.df.columns = self.df.columns[:-len(self.label_map)].append(pd.Index(self.label_map.values()))

    def _extension_check(self,
        df: pd.DataFrame) -> pd.DataFrame:
        """
        Check if filenames in DataRaw.df "id" column contain file extensions. If not, appends file extensions to
        filenames in DataRaw.df based on DataRaw.filetype. Returns a dataframe with (non-)edited "id" column.

        # Arguments:
            df: dataframe containing "id" column with image filenames/paths as the first column.
        """
        extensionsToCheck = ('jpg', 'png', 'tiff', 'jpeg', 'bmp', 'rgb',
                             'gif', 'pbm', 'pgm', 'ppm', 'rast', 'xbm')
        if str(df.iloc[0,0]).endswith(extensionsToCheck) == False:
            df.iloc[:,0] = df.iloc[:,0].map(lambda x: x + "." + self.filetype)
        return df

    def _prepend_parent_dirs(self,
        df: pd.DataFrame) -> pd.DataFrame:
        """
        Check if filenames in DataRaw.df "id" column contain full paths to file. If not, prepend the paths to
        filenames in DataRaw.df based on DataRaw.path. Returns a dataframe with (non-)edited "id" column.

        # Arguments:
            df: dataframe containing "id" column with image filenames/paths as the first column.
        """
        if ("/" not in df.iloc[0,0]) & ("\\" not in df.iloc[0,0]):
            df.iloc[:,0] = df.iloc[:,0].map(lambda x: self.path + "/train/" + str(x))
        return df


    def show_skewed_images(self,
        n_img: int = 3,
        tall: Union[float, int] = 2,
        wide: Union[float, int] = 0.6,
        cmap: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Retrieve and displays images in the DataRaw.df dataframe that have an unusually high or low
        dimension ratio (i.e. unusually tall, or unusually wide images).

        # Arguments:
            n_img: number of images to display.
            tall: ratio of height/width of an image (default: 2) that is considered unusually 'tall'.
            wide: ratio of height/width of an image (default: 0.6) that is considered unusually 'wide'.
            cmap: use custom matplotlib colormap for displaying images.
            figsize: size of the figure with displayed images.
        """
        df = self.df.copy()
        if "height" not in df.columns:
            df["height"] = df["id"].map(partial(utils.get_image_resolution, dim=0))
            df["width"] = df["id"].map(partial(utils.get_image_resolution, dim=1))
        df_copy = pd.DataFrame(
            df.loc[(df["height"]/df["width"] > 2) |
                        (df["height"]/df["width"] < 0.6), "id"]
        )
        if len(df_copy) > 0:
            print(f"Found {df_copy} skewed images.")
            df_copy["ratio"] = df["height"]/df["width"]
            df_copy.sort_values("ratio", inplace=True)
            df_copy.reset_index(inplace=True, drop=True)

            plt.figure(figsize=figsize)
            subplot_index = 1

            for img_index in [i for j in (range(n_img), range(len(df_copy)-n_img,len(df_copy))) for i in j]:
                plt.subplot(n_img-1,3,subplot_index)
                image_string = tf.io.read_file(df_copy.loc[img_index, "id"])
                image = tf.image.decode_jpeg(image_string, channels=3)
                plt.imshow(image, cmap=cmap)
                subplot_index += 1
        else:
            print("No skewed images found.")


    def show_images(self,
        df: pd.DataFrame = None,
        n_img: int = 3,
        cmap: str = None,
        figsize: Tuple[int, int] = (10,6)) -> None:
        """
        Display an example of random images from DataRaw.df.

        # Arguments
            n_img: number of images to display.
            cmap: use custom matplotlib color scheme for displaying images.
            figsize: size of the figure with displayed images.
        """
        if type(df) is not pd.DataFrame:
            select_df = self.df.sample(n=n_img)
        else:
            select_df = df.sample(n=n_img)
        fig, ax = plt.subplots(math.ceil(n_img/3), 3, figsize=figsize, squeeze=False,
            gridspec_kw={'hspace': 0.2})
        ax = ax.ravel()
        for i in range(n_img):
            img = tf.image.decode_jpeg(tf.io.read_file(select_df.iloc[i,0]))
            ax[i].imshow(img)
            ax[i].set_title(select_df.iloc[i,1])
        if n_img % 3 != 0:
            for i in range((n_img % 3)+1):
                ax.flat[-1-i].set_visible(False)
        plt.show()


    def rescale_input_images(self,
        new_dir: str,
        data_dir: Path,
        scale_size: float = 1.,
        dims: Optional[Tuple[int, int]] =(None, None)):
        """
        Downscale images in the train folder to make training faster.
        This function takes all train images, rescales them, puts them to "train"
        directory, and moves the original images to "new_dir" directory.

        # Arguments:
            new_dir: directory name, which will store the original unscaled images
            scale_size: multiplier for image height and with (> 1 = upscaling,
                < 1 = downscaling, 0.5 = shrink images by a half)
            dims: alternatively, specify new dimensions separately in absolute
                numbers. WARNING: if new dimension ratio differs from the original,
                it might skew the images.
        """
        self.df.reset_index(inplace=True, drop=True)
        os.mkdir(data_dir/"train_temp")
        os.mkdir(data_dir/new_dir)
        if "height" not in self.df.columns:
            self.df["height"] = self.df["id"].map(partial(utils.get_image_resolution, dim=0))
            self.df["width"] = self.df["id"].map(partial(utils.get_image_resolution, dim=1))
        pool = mp.Pool(mp.cpu_count())
        pool.map(partial(tf_resizer, scale_size=scale_size, dims=dims),
                      [self.df.iloc[i] for i in range(len(self.df))])
        pool.close()
        os.rename(data_dir/"train", data_dir/new_dir)
        os.rename(data_dir/"train_temp", data_dir/"train")

    def describe(self) -> None:
        """
        Print out descriptive statistics of the DataRaw.df.
        """
        if not self.y_multilabel:
            target_sums = self.df.groupby("label")["id"].count().sort_values(ascending=False)
            print(f"The data contain {len(self.df)} examples of {len(self.df.label.unique())} categories. The highest/lowest category counts are",
                f"\n'{self.label_map[target_sums.head(1).index.item()]}': {target_sums.iloc[0]} and",
                f"'{self.label_map[target_sums.tail(1).index.item()]}': {target_sums.iloc[-1]}")
            plt.figure(figsize=(10,10))
            plt.xticks(rotation=45)
            plt.title("Label distribution")
            sns.barplot(y=target_sums.index, x=target_sums, orient='h')
            plt.show()

            self.show_images(n_img=6)

        if self.y_multilabel:
            target = self.df[self.df.columns[~self.df.columns.isin(["id", "label", "img_id", "height", "width"])]]
            target_sums = target.apply(np.sum, axis=0).sort_values(ascending=False)
            print(f"The data contain {len(self.df)} examples of {len(self.label_map)} categories. The highest/lowest category counts are",
            f"{target_sums.index.iloc[0]}': {target_sums.iloc[0]} and '{target_sums.index.iloc[-1]}: {target_sums.iloc[-1]}")
            plt.figure(figsize=(10,10))
            plt.xticks(rotation=45)
            plt.title("Label distribution")
            sns.barplot(y=target_sums.index, x=target_sums, orient='h')
            plt.show()
            print(f"\n\nLabel per image stats:\n{self.df.label.map(len).describe()[[1,3,4,5,6,7]]}")
            n_cats = self.df.loc[:, "label"].map(len)
            target_multicats = target.loc[n_cats > 1, target.columns != "n_cats"].corr()
            plt.title("Category co-occurence")
            sns.heatmap(target_multicats, vmin=-1, vmax=1)
            print("\n")
            plt.show()
