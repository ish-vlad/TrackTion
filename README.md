# TrackTion
Разработка системы наблюдения за людьми по распределённому набору видеокамер с использованием технологий машинного зрения

# Darknet

Откуда брал код:
1) https://cloudxlab.com/blog/setup-yolo-with-darknet/
2) https://cloudxlab.com/blog/object-detection-yolo-and-python-pydarknet/

## Darknet repo

При необходимости:

```
git clone https://github.com/pjreddie/darknet
```

Затем все файлы можно будет скачать или так:
```python
import urllib.request

url = 'https://pjreddie.com/media/files/yolov3.weights' # Необходиый URL
filename = 'yolo3.weights.txt'

urllib.request.urlretrieve(url, filename)
```

Или так:
```python 
wget https://pjreddie.com/media/files/yolov3.weights
```
