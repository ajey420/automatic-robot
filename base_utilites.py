import base64

def format_path(path) : 

    path = path.replace('\n' , '')
    path = path.replace('\t' , '')
    path = path.replace(' ' , '')

    return path

def get_base64(bin_file) : 

    with open(bin_file , 'rb') as f : data = f.read()
    
    return base64.b64encode(data).decode()
