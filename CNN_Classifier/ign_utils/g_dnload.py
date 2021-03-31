import gdown

url = 'https://drive.google.com/drive/folders/1LvTyR6KYTZ6P972OWwDa45BL8ZAxRToD?usp=sharing'
output = '20150428_collected_images.tgz'
gdown.download(url, output, quiet=False)

#md5 = '3a7ca8eddcf70c08bb90c51ca4054817'
#gdown.cached_download(url, output, md5=md5, postprocess=gdown.extractall)

