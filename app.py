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
from textblob import TextBlob
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
from nltk.sentiment import SentimentIntensityAnalyzer
import io
from flask import Flask, render_template, request, jsonify

import nltk
nltk.download('vader_lexicon')

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
df = pd.read_csv('zaibten_scrap_datafile.csv')


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
        return f"{user['name']}!"
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

    sia = SentimentIntensityAnalyzer()  # Initialize SentimentIntensityAnalyzer

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

    # Create DataFrame with ratings and sentiment analysis
    data = pd.DataFrame({
        'Product Name': [product_name] * len(names),
        'Profile Name': names,
        'Rating': ratings,
        'Date': rating_dates,
        'Title': titles,
        'Review Text': reviews_text,
    })

    # Convert ratings to numeric for comparison
    data['Rating'] = data['Rating'].str.extract('(\d+\.\d+)').astype(float)  # Extract numeric rating

    # Set sentiment based on rating only
    data['Sentiment'] = data['Rating'].apply(lambda x: 'Negative' if x < 4 else 'Positive')

    # (Optional) Collect insights based on keywords but do not change sentiment
    for idx, row in data.iterrows():
        review_text = row['Review Text']
        if any(keyword in review_text.lower() for keyword in KEYWORDS['good']):
            # This is for insights; you can log or print for further analysis
            pass

    total_good_reviews = len(keywords_found['good'])
    total_bad_reviews = len(keywords_found['bad'])
    product_status = "Good Product: Recommended" if total_good_reviews > total_bad_reviews else "Not Recommended Product"

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
                                    tables=[data.to_html(classes='data', index=False)],  # Exclude index
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

# Function to determine sentiment
def Yelp_get_sentiment_score(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to scrape reviews data
def Yelp_scrape_reviews(url, product_name):
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(3)
    reviews_data = []

    while True:
        reviews_section = driver.find_elements(By.CSS_SELECTOR, "li.fdbk-container")
        for review in reviews_section:
            try:
                username = review.find_element(By.CSS_SELECTOR, "div.fdbk-container__details__info__username span").text
                date = review.find_element(By.CSS_SELECTOR, "span.fdbk-container__details__info__divide__time span").text
                comment = review.find_element(By.CSS_SELECTOR, "div.fdbk-container__details__comment span").text.strip()
                if not comment:
                    continue
                feedback_type_icon = review.find_element(By.CSS_SELECTOR, "div.fdbk-container__details__info__icon svg")
                feedback_type = feedback_type_icon.get_attribute("data-test-type")

                sentiment_score = Yelp_get_sentiment_score(comment)

                reviews_data.append({
                    "Product Name": product_name,
                    "Username": username,
                    "Date": date,
                    "Comment": comment,
                    "Feedback Type": feedback_type,
                    "Sentiment Score": feedback_type,
                })
            except Exception as e:
                print("An error occurred:", e)

        try:
            next_button = driver.find_element(By.CSS_SELECTOR, "a.pagination__next")
            next_href = next_button.get_attribute("href")
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(3)
            if next_button.get_attribute("href") == next_href:
                break
        except:
            break

    driver.quit()
    return reviews_data

# Function to create PDF with review analysis summary and new graphs
def Yelpcreate_pdf(summary, reviews_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)

    # Title
    pdf.cell(200, 10, txt="Review Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Summary Information
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Product Name: {summary['product_name']}", ln=True)
    pdf.cell(200, 10, txt=f"Good Reviews: {summary['good_reviews']}", ln=True)
    pdf.cell(200, 10, txt=f"Bad Reviews: {summary['bad_reviews']}", ln=True)
    pdf.cell(200, 10, txt=f"Neutral Reviews: {summary['neutral_reviews']}", ln=True)
    pdf.cell(200, 10, txt=f"Recommendation: {summary['recommendation']}", ln=True)
    pdf.ln(10)

    # Plot Good vs Bad Reviews
    plt.figure(figsize=(6, 4))
    labels = ['Good Reviews', 'Bad Reviews']
    values = [summary['good_reviews'], summary['bad_reviews']]
    plt.bar(labels, values, color=['green', 'red'])
    plt.title('Good vs Bad Reviews')
    img_io1 = BytesIO()
    plt.savefig(img_io1, format='png')
    img_io1.seek(0)
    plt.close()
    pdf.image(img_io1, x=10, y=pdf.get_y(), w=100)
    pdf.ln(50)

    # Plot Rating Distribution
    sentiment_counts = reviews_df['Sentiment Score'].value_counts()
    sentiment_counts = sentiment_counts.reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)  # Ensures all categories are present
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color='skyblue')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    img_io2 = BytesIO()
    plt.savefig(img_io2, format='png')
    img_io2.seek(0)
    plt.close()
    pdf.image(img_io2, x=10, y=pdf.get_y(), w=100)
    pdf.ln(50)

    # Plot Review Word Count Distribution
    word_counts = reviews_df['Comment'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(6, 4))
    plt.hist(word_counts, bins=20, color='purple', edgecolor='black')
    plt.title('Review Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    img_io3 = BytesIO()
    plt.savefig(img_io3, format='png')
    img_io3.seek(0)
    plt.close()
    pdf.image(img_io3, x=10, y=pdf.get_y(), w=100)
    pdf.ln(50)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# Function to send email with PDF attachment
def Yelpsend_email(pdf_attachment, summary):
    sender_email = "muzamilkhanofficial786@gmail.com"
    recipient_email = get_logged_in_user_email()
    password = "iaqu xvna tpix ugkt"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"Review Analysis Report for {summary['product_name']}"

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

@app.route('/ebay_index', methods=['GET', 'POST'])
@login_required
def ebay_Index():
    # Check if user is logged in
    if 'user_id' in session:
        user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
        username = user['name']  # Assuming the user's name is stored in the database
    else:
        username = None

    reviews_df = None
    positive_comments = []  # List for positive comments
    negative_comments = []  # New list for negative comments
    summary = {
        "good_reviews": 0,
        "bad_reviews": 0,
        "neutral_reviews": 0,
        "recommendation": "",
        "product_name": ""
    }

    if request.method == 'POST':
        product_name = request.form['product_name']
        url = request.form['url']
        summary["product_name"] = product_name
        reviews_data = Yelp_scrape_reviews(url, product_name)
        reviews_df = pd.DataFrame(reviews_data)

        # Filter positive comments
        positive_comments = reviews_df[reviews_df['Feedback Type'] == "positive"]['Comment'].tolist()

        # Filter negative comments
        negative_comments = reviews_df[reviews_df['Feedback Type'] == "negative"]['Comment'].tolist()

        # Count the types of reviews based on Feedback Type
        summary['good_reviews'] = reviews_df[reviews_df['Feedback Type'] == "positive"].shape[0]
        summary['bad_reviews'] = reviews_df[reviews_df['Feedback Type'] == "negative"].shape[0]
        summary['neutral_reviews'] = reviews_df[reviews_df['Feedback Type'] == "neutral"].shape[0]

        # Determine recommendation
        if summary['good_reviews'] > (summary['good_reviews'] + summary['bad_reviews'] + summary['neutral_reviews']) / 2:
            summary['recommendation'] = "Recommended"
        else:
            summary['recommendation'] = "Not Recommended"

        # Generate PDF and send email
        pdf_attachment = Yelpcreate_pdf(summary, reviews_df)
        Yelpsend_email(pdf_attachment, summary)

    return render_template(
        'ebay_index.html',
        tables=[reviews_df.to_html(classes='table table-striped')] if reviews_df is not None else None,
        summary=summary,
        username=username,
        positive_comments=positive_comments,  # Pass the list of positive comments
        negative_comments=negative_comments   # Pass the list of negative comments
    )

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



@app.route('/about', methods=['GET', 'POST'])
def about():
    # Check if user is logged in
    if 'user_id' in session:
        user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
        username = user['name']  # Assuming the user's name is stored in the database
    else:
        username = None
        
    return render_template("about.html", username=username)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    # Check if user is logged in
    if 'user_id' in session:
        user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
        username = user['name']  # Assuming the user's name is stored in the database
    else:
        username = None
        
    return render_template("contact.html", username=username)

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Clear the session
    return redirect(url_for('login'))

@app.route('/reviews', methods=['GET', 'POST'])
@login_required
def reviews():
    # Check if user is logged in
    if 'user_id' in session:
        user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
        username = user['name']  # Assuming the user's name is stored in the database
    else:
        username = None


    # Fetch unique values from 'category', 'subcategory', and 'model'
    categories = df['category'].dropna().unique()
    subcategories = df['subcategory'].dropna().unique()
    models = df['model'].dropna().unique()

    selected_category = None
    selected_subcategory = None
    selected_model = None

    best_product = None  # Initialize variable to hold best product details
    graph_url = None  # Initialize variable to hold the graph image URL
    pie_chart_url = None  # Initialize variable to hold the pie chart image URL
    histogram_url = None  # Initialize variable to hold histogram image URL
    boxplot_url = None  # Initialize variable to hold boxplot image URL
    scatterplot_url = None  # Initialize variable to hold scatterplot image URL
    line_chart_url = None  # Initialize variable to hold line chart image URL
    line_chart_url_2 = None  # Initialize variable for the second line chart
    pie_chart_url_2 = None  # Initialize variable for the second pie chart

    if request.method == 'POST':
        selected_category = request.form.get('category')
        selected_subcategory = request.form.get('subcategory')
        selected_model = request.form.get('model')

        # Filter the data based on user selection
        filtered_df = df[(
            df['category'] == selected_category) & 
            (df['subcategory'] == selected_subcategory) & 
            (df['model'] == selected_model)
        ]


        # Find the product with the highest rating
        if not filtered_df.empty:
            best_product = filtered_df.loc[filtered_df['rating'].idxmax()]  # Get row with max rating

            # Generate a bar chart for the ratings distribution
            rating_counts = filtered_df['rating'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(rating_counts.index, rating_counts.values, color='skyblue')
            ax.set_xlabel('Ratings')
            ax.set_ylabel('Number of Reviews')
            ax.set_title('Ratings Distribution')

            # Save the bar chart to a string in base64 format
            img_io = io.BytesIO()
            fig.savefig(img_io, format='png')
            img_io.seek(0)
            graph_url = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Generate a pie chart for the rating proportions
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'orange', 'lightcoral', 'gold'])
            ax.set_title('Rating Proportions')

            img_io = io.BytesIO()
            fig.savefig(img_io, format='png')
            img_io.seek(0)
            pie_chart_url = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Generate a histogram for review length distribution
            review_lengths = filtered_df['review_text'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(review_lengths, bins=20, color='skyblue', edgecolor='black')
            ax.set_xlabel('Review Length (Characters)')
            ax.set_ylabel('Frequency')
            ax.set_title('Review Length Distribution')

            img_io = io.BytesIO()
            fig.savefig(img_io, format='png')
            img_io.seek(0)
            histogram_url = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Generate a boxplot for rating distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot(filtered_df['rating'], patch_artist=True, boxprops=dict(facecolor='skyblue', color='black'))
            ax.set_ylabel('Ratings')
            ax.set_title('Boxplot of Ratings')

            img_io = io.BytesIO()
            fig.savefig(img_io, format='png')
            img_io.seek(0)
            boxplot_url = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Generate a scatter plot for review length vs. rating
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(review_lengths, filtered_df['rating'], color='blue', alpha=0.5)
            ax.set_xlabel('Review Length (Characters)')
            ax.set_ylabel('Ratings')
            ax.set_title('Review Length vs. Rating')

            img_io = io.BytesIO()
            fig.savefig(img_io, format='png')
            img_io.seek(0)
            scatterplot_url = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Generate a line chart for average ratings over time
            dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
            avg_ratings = [3.5, 4.0, 4.5, 3.0, 3.8, 4.2, 4.0, 3.6, 4.1, 4.3]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(dates, avg_ratings, marker='o', color='blue')
            ax.set_xlabel('Review Date')
            ax.set_ylabel('Average Rating')
            ax.set_title('Average Ratings Over Time')

            img_io = io.BytesIO()
            fig.savefig(img_io, format='png')
            img_io.seek(0)
            line_chart_url = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Generate a second line chart for rating changes over time
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(dates, [4.2, 4.1, 4.3, 3.9, 4.0, 4.1, 4.4, 4.0, 3.8, 4.2], marker='o', color='green')
            ax.set_xlabel('Review Date')
            ax.set_ylabel('Rating Value')
            ax.set_title('Rating Trend Over Time')

            img_io = io.BytesIO()
            fig.savefig(img_io, format='png')
            img_io.seek(0)
            line_chart_url_2 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Generate a second pie chart for sentiment distribution
            sentiment_counts = {'Positive': 80, 'Negative': 15, 'Neutral': 5}  # Example counts
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=['lightgreen', 'orange', 'lightcoral'])
            ax.set_title('Review Sentiment Distribution')

            img_io = io.BytesIO()
            fig.savefig(img_io, format='png')
            img_io.seek(0)
            pie_chart_url_2 = base64.b64encode(img_io.getvalue()).decode('utf-8')


    return render_template(
        'reviews.html',
        categories=categories,
        subcategories=subcategories,
        models=models,
        selected_category=selected_category,
        selected_subcategory=selected_subcategory,
        selected_model=selected_model,
        best_product=best_product,  # Pass best product to the template
        graph_url=graph_url,  # Pass the ratings distribution graph URL
        pie_chart_url=pie_chart_url,  # Pass the pie chart URL
        histogram_url=histogram_url,  # Pass the histogram URL
        boxplot_url=boxplot_url,  # Pass the boxplot URL
        scatterplot_url=scatterplot_url,  # Pass the scatter plot URL
        line_chart_url=line_chart_url,  # Pass the line chart URL
        line_chart_url_2=line_chart_url_2,  # Pass the second line chart URL
        pie_chart_url_2=pie_chart_url_2,  # Pass the second pie chart URL
        username=username
    )

@app.route('/get_subcategories/<category>', methods=['GET'])
def get_subcategories(category):
    subcategories = df[df['category'] == category]['subcategory'].dropna().unique()
    return jsonify(subcategories=subcategories.tolist())

@app.route('/get_models/<subcategory>', methods=['GET'])
def get_models(subcategory):
    models = df[df['subcategory'] == subcategory]['model'].dropna().unique()
    return jsonify(models=models.tolist())




if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')


# if __name__ == '__main__':
#     app.run(debug=True)


