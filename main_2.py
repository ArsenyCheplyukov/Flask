import base64
import os
import random
import time

import numpy as np
import undetected_chromedriver as uc
from fake_useragent import UserAgent
from fp.fp import FreeProxy
from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def random_user_agent():
    ua = UserAgent()
    return ua.random


def download_file_content_chrome(driver, uri, image_name="out_image.jpg"):
    result = driver.execute_async_script(
        """
        var uri = arguments[0];
        var callback = arguments[1];
        var toBase64 = function(buffer){for(var r,n=new Uint8Array(buffer),t=n.length,a=new Uint8Array(4*Math.ceil(t/3)),i=new Uint8Array(64),o=0,c=0;64>c;++c)i[c]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".charCodeAt(c);for(c=0;t-t%3>c;c+=3,o+=4)r=n[c]<<16|n[c+1]<<8|n[c+2],a[o]=i[r>>18],a[o+1]=i[r>>12&63],a[o+2]=i[r>>6&63],a[o+3]=i[63&r];return t%3===1?(r=n[t-1],a[o]=i[r>>2],a[o+1]=i[r<<4&63],a[o+2]=61,a[o+3]=61):t%3===2&&(r=(n[t-2]<<8)+n[t-1],a[o]=i[r>>10],a[o+1]=i[r>>4&63],a[o+2]=i[r<<2&63],a[o+3]=61),new TextDecoder("ascii").decode(a)};
        var xhr = new XMLHttpRequest();
        xhr.responseType = 'arraybuffer';
        xhr.onload = function(){ callback(toBase64(xhr.response)) };
        xhr.onerror = function(){ callback(xhr.status) };
        xhr.open('GET', uri);
        xhr.send();
        """,
        uri,
    )
    if type(result) == int:
        raise Exception("Request failed with status %s" % result)
    with open(image_name, "wb") as f:
        f.write(base64.b64decode(result))


def add_noise(image_name, number, start_folder, end_folder, noise_strength):
    # Open the image using PIL
    image = Image.open(os.path.join(start_folder, image_name))

    # Convert the image to a numpy array
    img_array = np.array(image)

    # Generate random noise with the same shape as the image
    noise = np.random.normal(0, noise_strength, img_array.shape)

    # Add the noise to the image array
    noisy_img_array = img_array + noise

    # Clip the pixel values to the valid range of 0-255
    noisy_img_array = np.clip(noisy_img_array, 0, 255)

    # Convert the array back to PIL image
    noisy_image = Image.fromarray(noisy_img_array.astype(np.uint8))

    # Save the noisy image
    noisy_image.save(os.path.join(end_folder, f"{image_name[:-4]}_{number}.jpg"))

    return os.path.join(end_folder, f"{image_name[:-4]}_{number}.jpg")


if __name__ == "__main__":
    start_file_path = "files"
    end_file_path = "modified_files"
    list_of_files = [f for f in os.listdir(start_file_path) if f.endswith(".jpg")]

    for i in list_of_files:
        proxy = FreeProxy().get()
        options = uc.ChromeOptions()
        options.add_argument(f"--proxy-server={proxy[7:]}")
        # options.add_argument(f"user-agent={random_user_agent()}")  # Add a random user agent

        driver = uc.Chrome(options=options)

        for j in range(10):
            noise_strength = np.random.randint(5, 25, 1)
            image_path = os.path.abspath(add_noise(i, j, start_file_path, end_file_path, noise_strength))

            driver.get("https://huggingface.co/lambdalabs/sd-image-variations-diffusers")

            try:
                file_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'input[type="file"][accept="image/*"][style*="display: none;"]')
                    )
                )
                # Introduce a random delay between requests
                time.sleep(random.uniform(0.2, 1))
                # Modify the element's style attribute to make it visible
                driver.execute_script('arguments[0].style.display = "block";', file_input)
                # Introduce a random delay between requests
                time.sleep(random.uniform(0.2, 1))
                # Upload the image by sending the file path to the file input element
                file_input.send_keys(image_path)
                # Introduce a random delay between requests
                time.sleep(random.uniform(0.2, 1.5))
                driver.find_element(By.XPATH, "//button[contains(text(), 'Compute')]").click()

                image_element = WebDriverWait(driver, 60).until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            '//*[contains(concat( " ", @class, " " ), concat( " ", "object-contain", " " ))]',
                        )
                    )
                )

                image_url = image_element.get_attribute("src")
                download_file_content_chrome(driver, image_url, image_path)
                print(f"Image saved successfully as {image_path}")
            except Exception as e:
                print(f"An error occurred while processing the image: {e}")

            # Introduce a random delay between requests
            time.sleep(random.uniform(1, 3))

        driver.quit()
