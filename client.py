import requests
import argparse

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/hat_predict'


def predict_result(image_name):
    # Initialize image path
    image = open(image_name, 'rb').read()
    payload = {'image': image}
    # Submit the request.
    try:
        r = requests.post(PyTorch_REST_API_URL, files=payload)
        r = r.json()
        # Ensure the request was successful.
        if r['success']:
            # Loop over the predictions and display them.
            print('response:', r)
        # Otherwise, the request failed.
        else:
            print('Request failed!')
    except:
        print("Server error!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', type=str, default='data/samples/1234.jpg', help='test image file')
    args = parser.parse_args()
    predict_result(args.file)
