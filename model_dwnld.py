from googledriver import download

URL = 'https://drive.google.com/file/d/15MODMyLk4fYPK5gKpnOYl88dagmEz4bV/view?usp=sharing'
cached_path = download(URL, None, 'tf_model')
