from functools import wraps
from bson import ObjectId
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import pymongo
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
import numpy as np
from datetime import datetime
import matplotlib
import os
from urllib.parse import urljoin
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
from fpdf import FPDF  # PDF generation library
matplotlib.use('Agg')
from bson import ObjectId
matplotlib.use('Agg')  # Use a non-interactive backend


app = Flask(__name__)

# Define keywords to look for in reviews
KEYWORDS = {
    'good': ['good', 'great', 'excellent', 'fantastic', 'love', 'like', 'satisfied', 'amazing', '❤️', 'Thankyou'],
    'bad': ['bad', 'poor', 'terrible', 'disappointed', 'hate', 'worse', 'not good', 'Fake', '😡', 'not great'],
    'recommended': ['recommend', 'recommended', 'suggest', 'advise', 'should buy']
}

# Initialize WebDriver outside the routes
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
#driver.get("https://www.amazon.com")

# Load the CSV file
#df = pd.read_csv('Iphone 13 Amazon Reviews.csv')
df = pd.read_csv('Amazon_Mobile_Reviews.csv')


# Print column names for debugging
print(df.columns)

# Replace 'category' with the actual column name
categories = df['category'].unique()

def create_review_plot(good_count, bad_count):
    labels = ['Good Reviews', 'Bad Reviews']
    sizes = [good_count, bad_count]
    colors = ['#4CAF50', '#F44336']
    explode = (0.1, 0)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_rating_distribution_plot(filtered_df):
    filtered_df.loc[:, 'rating_numeric'] = pd.to_numeric(filtered_df['rating'].str.split(' ').str[0], errors='coerce')
    rating_counts = filtered_df['rating_numeric'].value_counts().sort_index()
    
    plt.figure(figsize=(8, 5))
    rating_counts.plot(kind='bar', color='skyblue')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=0)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_average_rating_plot(filtered_df):
    average_rating = filtered_df['rating_numeric'].mean()
    
    plt.figure(figsize=(6, 4))
    plt.bar(['Average Rating'], [average_rating], color='lightblue')
    plt.ylim(0, 5)
    plt.title('Average Rating')
    plt.ylabel('Rating')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_box_plot(filtered_df):
    plt.figure(figsize=(6, 4))
    plt.boxplot(filtered_df['rating_numeric'], vert=False)
    plt.title('Box Plot of Ratings')
    plt.xlabel('Rating')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_word_cloud(filtered_df):
    reviews = ' '.join(filtered_df['review_text'].dropna().astype(str))
    if not reviews.strip():
        return None  # No review text, return None or some placeholder
   
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_top_reviews_plot(filtered_df):
    filtered_df['rating_numeric'] = pd.to_numeric(filtered_df['rating'].str.split(' ').str[0], errors='coerce')
    top_reviews = filtered_df.dropna(subset=['rating_numeric']).nlargest(10, 'rating_numeric')
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(range(len(top_reviews)))  # Generate multiple colors
    plt.barh(top_reviews['product_name'], top_reviews['rating_numeric'], color=colors)
    plt.title('Top 10 Reviews')
    plt.xlabel('Rating')
    plt.ylabel('Product Name')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_top_reviewers_plot(filtered_df):
    top_users = filtered_df['profile_name'].value_counts().nlargest(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_users.index, y=top_users.values, palette='plasma')
    plt.title('Top 10 Reviewers')
    plt.xlabel('Reviewer')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_sentiment_analysis_plot(filtered_df):
    sentiment_counts = filtered_df['rating_numeric'].apply(lambda x: 'Positive' if x >= 4 else 'Negative').value_counts()
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='muted')
    plt.title('Sentiment Analysis of Reviews')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_trend_of_ratings_plot(filtered_df):
    if 'date' in filtered_df.columns:
        filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
        ratings_over_time = filtered_df.groupby(filtered_df['date'].dt.to_period('M'))['rating_numeric'].mean()
        
        plt.figure(figsize=(10, 5))
        ratings_over_time.plot(marker='o')
        plt.title('Trend of Average Ratings Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45)
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode()
    return None

def create_top_five_words_plot(filtered_df):
    reviews = ' '.join(filtered_df['review_text'].astype(str))
    words = [word for word in reviews.split() if len(word) > 2]  # Filter out short words
    most_common_words = Counter(words).most_common(5)
    
    words, counts = zip(*most_common_words)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(counts), y=list(words), palette='viridis')
    plt.title('Top 5 Most Frequent Words in Reviews')
    plt.xlabel('Count')
    plt.ylabel('Words')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_review_length_distribution_plot(filtered_df):
    filtered_df['review_length'] = filtered_df['review_text'].apply(lambda x: len(str(x)))
    
    plt.figure(figsize=(8, 5))
    sns.histplot(filtered_df['review_length'], bins=30, kde=True)
    plt.title('Distribution of Review Lengths')
    plt.xlabel('Length of Review (characters)')
    plt.ylabel('Frequency')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def create_ratings_heatmap(filtered_df):
    heatmap_data = filtered_df.pivot_table(index='subcategory', columns='rating_numeric', values='review_text', aggfunc='count', fill_value=0)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='Blues')
    plt.title('Ratings Heatmap Across Subcategories')
    plt.xlabel('Ratings')
    plt.ylabel('Subcategories')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()


app.secret_key = '05b49f8c793c97f1f870020fb69c4571fd2e5607'

# MongoDB connection
app.config["MONGO_URI"] = "mongodb+srv://dawoodzahid488:BozqArXOh5NImhu3@cluster0.nyf6y.mongodb.net/your_database?retryWrites=true&w=majority"
mongo = pymongo.MongoClient(app.config["MONGO_URI"])
db = mongo.Zaibten  # Replace 'your_database' with your actual database name
users_collection = db.users  # Replace with your users collection name


@app.route('/')
@app.route('/home')
def home():
    # Check if user is logged in
    if 'user_id' in session:
        user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
        username = user['name']  # Assuming the user's name is stored in the database
    else:
        username = None

    # Load the CSV file with reviews
    csv_file = 'combined_reviews.csv'  # Update the path to your CSV file
    df = pd.read_csv(csv_file)

    # Filter reviews with a 5-star rating
    df_filtered = df[df['rating'] == '5.0 out of 5 stars']

    # Drop duplicate products and select the first 9 unique ones
    df_unique_products = df_filtered.drop_duplicates(subset='product_name').head(12)

    # Prepare the data to pass to the template
    reviews = df_unique_products[['product_name', 'review_text', 'model', 'category', 'image']].to_dict(orient='records')

    # Render the home.html template, passing both the username and reviews data
    return render_template('home.html', username=username, reviews=reviews)



def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users_collection.find_one({"email": email, "password": password})

        if user:
            session['user_id'] = str(user['_id'])
            return redirect(url_for('home'))
        else:
            # Pass 'invalid' as True if login fails
            return render_template('login.html', invalid=True)
    
    # Default 'invalid' as False for GET requests
    return render_template('login.html', invalid=False)


# Route for signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            return 'User already exists'
        users_collection.insert_one({"name": name, "email": email, "password": password})
        return redirect(url_for('login'))
    return render_template('signup.html')

# Dashboard route (after login)

@app.route('/layout')
def dashboard():
    if 'user_id' in session:
        user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
        return f"Welcome {user['name']}!"
    else:
        return redirect(url_for('login'))

@app.route('/livescrap')
@login_required
def livescrap():
        # Check if user is logged in
    if 'user_id' in session:
        user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
        username = user['name']  # Assuming the user's name is stored in the database
    else:
        username = None

    return render_template('livescrap.html', username=username)

@app.route('/scrape', methods=['POST'])
@login_required
def scrape():
        # Check if user is logged in
    if 'user_id' in session:
        user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
        username = user['name']  # Assuming the user's name is stored in the database
    else:
        username = None

    product_name = request.form['product_name']
    url = request.form['product_url']

    names, ratings, rating_dates, titles, reviews_text, images = [], [], [], [], [], []
    keywords_found = {'good': [], 'bad': [], 'recommended': []}
    page_number = 1

    while True:
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@data-hook='review']"))
            )
            html_data = BeautifulSoup(driver.page_source, 'html.parser')
            reviews = html_data.find_all('div', {'data-hook': 'review'})
            if not reviews:
                break

            for review in reviews:
                name = review.find('span', {'class': 'a-profile-name'}).text.strip()
                names.append(name)
                rating = review.find('span', {'class': 'a-icon-alt'}).text
                ratings.append(rating)
                rating_date = review.find('span', {'data-hook': 'review-date'}).text
                rating_dates.append(rating_date)
                title = review.find('a', {'data-hook': 'review-title'}).text
                titles.append(title)
                review_text = review.find('span', {'data-hook': 'review-body'}).text
                reviews_text.append(review_text)

                for key in keywords_found.keys():
                    for keyword in KEYWORDS[key]:
                        if keyword in review_text.lower() and review_text not in keywords_found[key]:
                            keywords_found[key].append(review_text)

                images.append('')

            next_page = html_data.find('li', {'class': 'a-last'})
            if next_page and next_page.a and 'href' in next_page.a.attrs:
                url = urljoin(url, next_page.a['href'])
                page_number += 1
            else:
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    total_good_reviews = len(keywords_found['good'])
    total_bad_reviews = len(keywords_found['bad'])
    product_status = "Good Product: Recommended" if total_good_reviews > total_bad_reviews else "Not Recommended Product"

    
    data = pd.DataFrame({
        'Product Name': [product_name] * len(names),
        'Profile Name': names,
        'Rating': ratings,
        'Date': rating_dates,
        'Title': titles,
        'Review Text': reviews_text,
        # 'Image': images
    })

    # Plot good vs bad reviews
    plt.figure(figsize=(6, 4))
    labels = ['Good Reviews', 'Bad Reviews']
    values = [total_good_reviews, total_bad_reviews]
    plt.bar(labels, values, color=['green', 'red'])
    plt.title('Good vs Bad Reviews')
    img_io1 = BytesIO()
    plt.savefig(img_io1, format='png')
    img_io1.seek(0)
    plt.close()

    # Plot distribution of ratings
    rating_counts = pd.Series(ratings).value_counts()
    plt.figure(figsize=(6, 4))
    rating_counts.plot(kind='bar', color='skyblue')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    img_io2 = BytesIO()
    plt.savefig(img_io2, format='png')
    img_io2.seek(0)
    plt.close()

    # Plot word count analysis
    word_counts = [len(review.split()) for review in reviews_text]
    plt.figure(figsize=(6, 4))
    plt.hist(word_counts, bins=20, color='purple', edgecolor='black')
    plt.title('Review Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    img_io3 = BytesIO()
    plt.savefig(img_io3, format='png')
    img_io3.seek(0)
    plt.close()

    send_email([img_io1, img_io2, img_io3], product_name, product_status, total_good_reviews, total_bad_reviews)

    img_data = base64.b64encode(img_io1.getvalue()).decode()
    rendered_page = render_template('livescrap.html', 
                                    tables=[data.to_html(classes='data')],
                                    titles=['Scraped Reviews'],
                                    keywords_found=keywords_found,
                                    total_good_reviews=total_good_reviews,
                                    total_bad_reviews=total_bad_reviews,
                                    product_status=product_status,
                                    graph_data=img_data, username=username)
    return rendered_page


def create_pdf(img_ios, product_name, product_status, total_good_reviews, total_bad_reviews):
    pdf = FPDF()
    pdf.add_page()

    # Add logo
    # logo_path = 'static/images/logo.png'  # Replace with your logo's path
    # pdf.image(logo_path, 10, 8, 33)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Review Analysis for Product: {product_name}", ln=True, align="C")
    pdf.ln(10)

    # Product details
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Product Status: {product_status}", ln=True)
    pdf.cell(200, 10, txt=f"Total Good Reviews: {total_good_reviews}", ln=True)
    pdf.cell(200, 10, txt=f"Total Bad Reviews: {total_bad_reviews}", ln=True)
    pdf.ln(10)

    # Attach each graph
    for img_io in img_ios:
        pdf.image(img_io, x=10, w=180)
        pdf.ln(10)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

def get_logged_in_user_email():
    # Ensure 'user_id' is in session
    if 'user_id' not in session:
        print("No user ID in session.")
        return None

    try:
        # Fetch the currently logged-in user based on session ID
        user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
        if user:
            return user.get("email")  # Ensure your email field is correct
    except Exception as e:
        print(f"Error fetching user email: {e}")
    return None

def send_email(img_ios, product_name, product_status, total_good_reviews, total_bad_reviews):
    sender_email = "muzamilkhanofficial786@gmail.com"
    recipient_email = get_logged_in_user_email()
    
    if not recipient_email:
        print("No recipient email found; email not sent.")
        return  # Exit function if recipient email is None
    
    password = "iaqu xvna tpix ugkt"
    msg = MIMEMultipart('related')
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"Review Analysis for Product: {product_name}"

    # Create PDF and attach
    pdf_attachment = create_pdf(img_ios, product_name, product_status, total_good_reviews, total_bad_reviews)
    attach_part = MIMEBase('application', 'octet-stream')
    attach_part.set_payload(pdf_attachment.read())
    encoders.encode_base64(attach_part)
    attach_part.add_header('Content-Disposition', 'attachment', filename="Review_Analysis.pdf")
    msg.attach(attach_part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    selected_category = request.form.get('category')
    selected_subcategory = request.form.get('subcategory')
    selected_model = request.form.get('model')  # Added model dropdown
    subcategories = []
    models = []
    product_name = None
    good_review_count = 0
    bad_review_count = 0
    review_plot = None
    rating_distribution_plot = None
    average_rating_plot = None
    box_plot = None
    word_cloud = None
    top_reviews_plot = None
    top_reviewers_plot = None
    sentiment_plot = None
    trend_plot = None
    top_words_plot = None
    review_length_plot = None
    ratings_heatmap_plot = None

    if selected_category:
        subcategories = df[df['category'] == selected_category]['subcategory'].unique()
        
        if selected_subcategory:
            models = df[(df['category'] == selected_category) & 
                        (df['subcategory'] == selected_subcategory)]['model'].unique()  # Fetch models

            if selected_model:
                filtered_df = df[(df['category'] == selected_category) & 
                                 (df['subcategory'] == selected_subcategory) &
                                 (df['model'] == selected_model)]
                
                filtered_df['rating_numeric'] = pd.to_numeric(filtered_df['rating'].str.split(' ').str[0], errors='coerce')

                good_reviews = filtered_df[filtered_df['rating_numeric'] >= 4.0]
                if not good_reviews.empty:
                    product_name = good_reviews['product_name'].iloc[0]
                    good_review_count = len(good_reviews)
                    bad_review_count = len(filtered_df) - good_review_count
                    review_plot = create_review_plot(good_review_count, bad_review_count)
                    rating_distribution_plot = create_rating_distribution_plot(filtered_df)
                    average_rating_plot = create_average_rating_plot(filtered_df)
                    box_plot = create_box_plot(filtered_df)
                    word_cloud = create_word_cloud(filtered_df)
                    top_reviews_plot = create_top_reviews_plot(filtered_df)
                    top_reviewers_plot = create_top_reviewers_plot(filtered_df)
                    sentiment_plot = create_sentiment_analysis_plot(filtered_df)
                    trend_plot = create_trend_of_ratings_plot(filtered_df)
                    top_words_plot = create_top_five_words_plot(filtered_df)
                    review_length_plot = create_review_length_distribution_plot(filtered_df)
                    ratings_heatmap_plot = create_ratings_heatmap(filtered_df)

    return render_template('index.html', categories=categories, subcategories=subcategories, 
                           models=models,  # Pass models to template
                           selected_category=selected_category, selected_subcategory=selected_subcategory, 
                           selected_model=selected_model,  # Pass model to form
                           product_name=product_name, good_review_count=good_review_count, 
                           bad_review_count=bad_review_count, review_plot=review_plot,
                           rating_distribution_plot=rating_distribution_plot, average_rating_plot=average_rating_plot, 
                           box_plot=box_plot, word_cloud=word_cloud, top_reviews_plot=top_reviews_plot,
                           top_reviewers_plot=top_reviewers_plot, sentiment_plot=sentiment_plot,
                           trend_plot=trend_plot, top_words_plot=top_words_plot,
                           review_length_plot=review_length_plot, ratings_heatmap_plot=ratings_heatmap_plot)


@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Clear the session
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')


# if __name__ == '__main__':
#     app.run(debug=True)


