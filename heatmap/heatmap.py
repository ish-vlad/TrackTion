from typing import Optional
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

np.seterr(divide='ignore')
# plt.ioff()

class Heatmap:

    def __init__(self, path_to_csv, image_shift: int = 5):
        self.data = pd.read_csv(path_to_csv, index_col=0)

        self._extreme_point_x_max = self.data.box_xc_cm.max() + image_shift
        self._extreme_point_x_min = self.data.box_xc_cm.min() - image_shift
        self._extreme_point_y_max = self.data.box_yc_cm.max() + image_shift
        self._extreme_point_y_min = self.data.box_yc_cm.min() - image_shift

        self._xaxis_length = abs(self._extreme_point_x_max - self._extreme_point_x_min)
        self._yaxis_length = abs(self._extreme_point_y_max - self._extreme_point_y_min)

        self.data['scale_xy'] = self.data.apply(lambda x: self._coords_scaling(x['box_xc_cm'], x['box_yc_cm']), axis=1)

    def __len__(self):
        return self.data.shape[0]

    def _get_unique_id(self):
        return self.data.id.unique().tolist()

    def _get_unique_age(self):
        return self.data.age.unique().tolist()

    def _get_unique_gender(self):
        return self.data.gender.unique().tolist()

    def _coords_scaling(self, x, y):
        xsize = self._xaxis_length
        ysize = self._yaxis_length

        x1 = int(
            xsize / (self._extreme_point_x_max - self._extreme_point_x_min) * (self._extreme_point_x_max - x)) #TODO В данном случае из xsize вычитать ничего не нужно, ибо система координат построена в другую сторону
        y1 = ysize - int(
            ysize / (self._extreme_point_y_max - self._extreme_point_y_min) * (self._extreme_point_y_max - y))

        x_turn = y1
        y_turn = x1

        return x_turn, y_turn

    @staticmethod
    def __apply_filter(filter_values, initial_values):
        return filter_values if filter_values else initial_values

    def __create_mask(self, column, filter_value):
        return self.data[column].isin(filter_value)

    def __prepare_filter_data(self,
                              filter_gender: Optional[list] = None,
                              filter_age: Optional[list] = None,
                              filter_id: Optional[list] = None):
        ids = self._get_unique_id()
        ages = self._get_unique_age()
        genders = self._get_unique_gender()

        id_filter = self.__apply_filter(filter_id, ids)
        age_filter = self.__apply_filter(filter_age, ages)
        gender_filter = self.__apply_filter(filter_gender, genders)

        mask_id = self.__create_mask('id', id_filter)
        mask_age = self.__create_mask('age', age_filter)
        mask_gender = self.__create_mask('gender', gender_filter)

        data_filter = self.data[mask_id & mask_age & mask_gender]
        return data_filter

    def __plot_heatmap(self, x, y, smoothing, bins=1000):
        xlim = [0, self._yaxis_length]
        ylim = [0, self._xaxis_length]

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])

        extent = [0, int(self._yaxis_length / 2.5), 0, self._xaxis_length]

        logheatmap = np.log(heatmap)
        logheatmap[np.isneginf(logheatmap)] = 0
        logheatmap = ndimage.filters.gaussian_filter(logheatmap, smoothing, mode='nearest')
        logheatmap[logheatmap < 0.0001] = 0
        logheatmap = np.max(logheatmap) - logheatmap

        return logheatmap.T, extent

    @staticmethod
    def demographic_feature(data, column_features=['age', 'gender']):
        total_num_objects = data.id.nunique()
        result_dict = {'total_num_objects': total_num_objects}

        for column in column_features:
            df_current = data[['id', column]].drop_duplicates(). \
                groupby(column).agg({'id': 'count'}). \
                rename(columns={'id': f'num_of_{column}'})

            df_current['percent'] = (df_current[f'num_of_{column}'] / total_num_objects * 100).round(1)
            feature_dict = {column: df_current.to_dict('index')}
            result_dict = {**result_dict, **feature_dict}

        return result_dict

    def draw_heatmap_with_filters(self,
                                  smoothing: int = 30,
                                  filter_gender: Optional[list] = None,
                                  filter_age: Optional[list] = None,
                                  filter_id: Optional[list] = None
                                  ):
        """

        :param smoothing: Коэффициент размытия
        :param filter_gender: Или None, или список фильтров. Пример: ['Male']
        :param filter_age: Или None, или список фильтров. Пример: ['60+']
        :param filter_id: Или None, или список фильтров. Пример: ['1', '2']

        :return:
        1) fig (тепловая карта)
        2) json_info (статистика в формате json)
        """


        data = self.__prepare_filter_data(
            filter_gender,
            filter_age,
            filter_id
        )

        json_info = self.demographic_feature(data)

        scale_coords = data['scale_xy'].values

        scale_x = [x[0] for x in scale_coords]
        scale_y = [x[1] for x in scale_coords]

        fig = Figure(figsize=(20, 20))
        axis = fig.add_subplot(1, 1, 1)

        img, extent = self.__plot_heatmap(scale_x, scale_y, smoothing)
        axis.imshow(img, extent=extent, origin='lower', cmap="hot")
        axis.axis('off')

        return fig, json_info

