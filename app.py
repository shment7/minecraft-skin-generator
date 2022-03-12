from flask import Flask, send_file, redirect, render_template
import pickle
import torch
import PIL.Image
import numpy as np
import io
from random import randint 

app = Flask(__name__)

@app.route('/')
def home():
    i = randint(1, 1e+9)
    return render_template('home.html', seed=i)

@app.route('/<int:i>')
def get_image(i):
    seed = i
    with open('/home/minecraftskingenerator/mysite/minecraft.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema']
        
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))                              
    img = G(z, None, truncation_psi=0.8, force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/PNG')

    

