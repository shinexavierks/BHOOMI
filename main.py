from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename
import requests
import torch
from torchvision import transforms
from PIL import Image
import io
import pickle
import pandas as pd
import numpy as np
from utils.model import ResNet9
from markupsafe import Markup
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from flask_mail import Message, Mail

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farmers_buyers.db'

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(app)


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'roguealex444@gmail.com'
app.config['MAIL_PASSWORD'] = 'ukmpdbznvbgtbrkt'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    user_type = db.Column(db.String(10), nullable=False)  # 'farmer' or 'buyer'

class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    stock = db.Column(db.Integer, nullable=False)
    description = db.Column(db.Text, nullable=False)
    image = db.Column(db.String(200), nullable=False)  # Stores the image file path
    seller_id = db.Column(db.Integer, nullable=False)
    
    def get_seller(self):
        return User.query.get(self.seller_id)

class Cart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    item_id = db.Column(db.Integer, nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    
    def get_buyer(self):
        return User.query.get(self.user_id)

    def get_item(self):
        return Item.query.get(self.item_id)
    
    def get_seller(self):
        return self.get_item().get_seller()

@app.route('/')
def index():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if user.user_type == 'farmer':
        # Show only the items that belong to the logged-in farmer, including those with 0 stock
        items = Item.query.filter_by(seller_id=user.id).order_by(Item.stock.desc()).all()
    else:
        # Show all items in the market for buyers, excluding those with 0 stock
        items = Item.query.filter(Item.stock > 0).order_by(Item.stock.desc()).all()
    
    return render_template('index.html', items=items, user_type=user.user_type)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        user_type = request.form['user_type']
        hashed_password = generate_password_hash(password, method='sha256')
        user = User(username=username, email=email, password=hashed_password, user_type=user_type)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful!')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_type'] = user.user_type
            flash('Login successful!')
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your username and/or password.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_type', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/sell', methods=['GET', 'POST'])
def sell():
    if 'user_id' not in session or session['user_type'] != 'farmer':
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        name = request.form['name']
        price = float(request.form['price'])
        stock = int(request.form['stock'])
        description = request.form['description']
        
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            new_item = Item(name=name, price=price, stock=stock, description=description, image=image_path, seller_id=session['user_id'])
            db.session.add(new_item)
            db.session.commit()
            flash('Item added successfully!')
            return redirect(url_for('index'))
    
    return render_template('sell.html')

@app.route('/add_to_cart/<int:item_id>', methods=['POST'])
def add_to_cart(item_id):
    if 'user_id' not in session:
        return redirect(url_for('index'))
    quantity = int(request.form['quantity'])
    item = Item.query.get(item_id)
    if item and item.stock >= quantity:
        cart_item = Cart(user_id=session['user_id'], item_id=item_id, quantity=quantity)
        item.stock -= quantity
        db.session.add(cart_item)
        db.session.commit()
        flash('Item added to cart!')
    else:
        flash('Insufficient stock!')
    return redirect(url_for('index'))

@app.route('/checkout')
def checkout():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    cart_items = Cart.query.filter_by(user_id=session['user_id']).all()
    items = []
    total = 0
    farmers = []
    
    for cart_item in cart_items:
        item = cart_item.get_item()
        print(item.name)
        total += item.price * cart_item.quantity
        items.append({'name': item.name, 'quantity': cart_item.quantity, 'seller': item.get_seller().username, 'price': item.price})
        if cart_item.get_seller().id not in farmers:
            farmers.append(cart_item.get_seller().id)
    print(items)
    delivery_cost = 40 * len(farmers)
    total_with_delivery = total + delivery_cost
    
    return render_template('checkout.html', items=items, total=total, delivery_cost=delivery_cost, total_with_delivery=total_with_delivery)



@app.route('/payment', methods=['GET', 'POST'])
def payment():
    if request.method == 'POST':
        name = request.form['name']
        number = request.form['number']
        address = request.form['house']
        area = request.form['area']
        landmark = request.form['landmark']
        pincode = request.form['pincode']
        city = request.form['city']
        state = request.form['state']
        location = request.form['location']


        # Get cart items from session
        cart_items = Cart.query.filter_by(user_id=session['user_id']).all()

        # Calculate total amount and farmer details
        farmers = {}
        total = 0
        for cart_item in cart_items:
            item = Item.query.get(cart_item.item_id)
            total += item.price * cart_item.quantity
            if item.get_seller().email not in farmers:
                farmers[item.get_seller().email] = {
                    'items': [],
                    'total': 0,
                    'address': {
                        'name': name,
                        'number': number,
                        'address': address,
                        'area': area,
                        'landmark': landmark,
                        'pincode': pincode,
                        'city': city,
                        'state': state,
                        'location': location                    }
                }
            farmers[item.get_seller().email]['items'].append({
                'name': item.name,
                'price': item.price,
                'quantity': cart_item.quantity
            })
            farmers[item.get_seller().email]['total'] += item.price * cart_item.quantity

        # Send email to buyer
        buyerid = session['user_id']
        buyer = User.query.get(buyerid)
        buyer_email = buyer.email
        msg = Message("Order Confirmation", sender="roguealex444@gmail.com", recipients=[buyer_email])
        msg.body = f"""
        Thank you for your purchase!

        Total Amount: ₹{total}

        Delivery Address:
        {name}
        {number}
        {address}
        {area}
        {landmark}
        {pincode}
        {city}
        {state}

        Your items will be delivered to the provided address.
        """
        mail.send(msg)

        # Send email to farmers
        for email, details in farmers.items():
            msg = Message("New Order Received", sender="roguealex444@gmail.com", recipients=[email])
            items_list = '\n'.join([f"{item['name']} - ₹{item['price']} x {item['quantity']}" for item in details['items']])
            msg.body = f"""
            You have received a new order!

            Order Details:
            {items_list}
            Total Amount: ₹{details['total']}

            Delivery Address:
            {details['address']['name']}
            {details['address']['number']}
            {details['address']['address']}
            {details['address']['area']}
            {details['address']['landmark']}
            {details['address']['pincode']}
            {details['address']['city']}
            {details['address']['state']}
            Location: {details['address']['location']}
            """
            mail.send(msg)

        # Clear the cart after successful payment
        Cart.query.filter_by(user_id=session['user_id']).delete()
        db.session.commit()

        return redirect(url_for('index'))

    return render_template('payment.html')

@app.route('/update_item/<int:item_id>', methods=['GET', 'POST'])
def update_item(item_id):
    if 'user_id' not in session or session['user_type'] != 'farmer':
        return redirect(url_for('index'))
    item = Item.query.get(item_id)
    if request.method == 'POST' and item:
        item.price = float(request.form['price'])
        item.stock = int(request.form['stock'])
        db.session.commit()
        flash('Item updated successfully!')
        return redirect(url_for('index'))
    return render_template('update_item.html', item=item)

@app.route('/delete_item/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    if 'user_id' not in session or session['user_type'] != 'farmer':
        return redirect(url_for('index'))
    item = Item.query.get(item_id)
    if item:
        item.stock = 0
        db.session.commit()
        flash('Item stock set to 0 successfully!')
    return redirect(url_for('index'))

@app.route('/account')
def account():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    items_sold = Item.query.filter_by(seller_id=user.id).all() if user.user_type == 'farmer' else []
    cart_items = Cart.query.filter_by(user_id=user.id).all() if user.user_type == 'buyer' else []
    orders=True if cart_items!=[] else False
    return render_template('account.html', user=user, items=items_sold, cart_items=cart_items, orders=orders)



#======================================================================================

@app.route('/tools')
def tools():
    title = 'BHOOMI - Tools'
    return render_template('tools.html', title=title)


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations

weather_api_key = ""

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = weather_api_key
    url=f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=5&appid={api_key}"
    response = requests.get(url)
    x = response.json()
    lat=x[0]['lat']
    lon=x[0]['lon']
    gurl=f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(gurl)
    x = response.json()

    if x["cod"] != "404":
        y=x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'bhoomi - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'bhoomi - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'bhoomi - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('models/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

@app.route('/crop-recommend')
def crop_recommend():
    title = 'bhoomi - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page

@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'bhoomi - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

import pandas as pd
df = pd.read_csv('price.csv')

data_list = df.values.tolist()
@app.route('/price',methods=['GET','POST'])
def price():
    title='bhoomi - Price checker'
    products=[]
    for data in data_list:
        if data[10] not in products:
            products.append(data[10])
    if request.method=='POST':
        crop=request.form.get('product')
        state=request.form.get('state')
        district=request.form.get('city')
        pred=[]
        for data in data_list:
            if data[0]==state and data[1]==district and data[10]==crop:
                pred.append(data)
        return render_template('pricepred.html',title=title,pred=pred[0])
    return render_template('price.html',title=title,products=products)

def forecaster(city_name):
    api_key = weather_api_key
    url=f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=5&appid={api_key}"
    response = requests.get(url)
    x = response.json()
    lat=x[0]['lat']
    lon=x[0]['lon']
    gurl=f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(gurl)
    return response.json()['list']

@app.route('/weather',methods=['GET','POST'])
def weather():
    title='bhoomi - Weather forecast'
    if request.method=='POST':
        city=request.form.get('city')
        response=forecaster(city)
        return render_template('weather.html',title=title,pred=response)
    return render_template('weather.html',title=title)

def aic(text):
    url = "https://chatgpt-42.p.rapidapi.com/gpt4"

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"only answer agriculture related question, else just say question isn't agriculture related. question is {text}"
            }
        ],
        "web_access": False
    }
    headers = {
        "x-rapidapi-key": "8a18d19c93msh09e99cc221197d0p148c57jsna345c9180ccd",
        "x-rapidapi-host": "chatgpt-42.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    string= response.json()['result']
    return string

@app.route('/assistant',methods=['GET','POST'])
def assistant():
    title='bhoomi - Assistant'
    if request.method=='POST':
        query=request.form.get('query')
        response=aic(query)
        return render_template('assist.html',title=title,text=response)
    return render_template('assist.html',title=title,text="")

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
