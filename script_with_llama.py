import random
import pandas as pd
import requests
from gnews import GNews
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


# Initialize the GNews client
# API could be changed so make sure this is a correct import
gnews_client = GNews(language='en')

# Importing keys
def get_file_contents(filename):
    """ Given a filename,
        return the contents of that file
    """
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)
        
llamaKey = get_file_contents('keys/hugging_face.txt')
access_token = llamaKey
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)

model = AutoModelForCausalLM.from_pretrained(
    model, 
    token=access_token
)

llama_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_news_articles(query):
    """Fetch news articles using the GNews library."""
    articles = gnews_client.get_news(query)
    return articles

# Ask Llama regarding the pooled articles
def analyze_relationships(text):
    """Analyze company relationships using the Llama 2 model from Hugging Face."""
    prompt = (f"Process all these separate articles and return a single list of categorizations: {text}\n\n"
              "What are the companies mentioned in the article? For every company pair, "
              "assign their relationship to one of the following types: "
              "Collaboration, Cooperative, Coordination, Adversarial, Transactional, Competition, Coopetitive")
    response = llama_pipeline(prompt, max_length=512)
    return response[0]['generated_text']
def main():
    query = input("Enter the search query for news articles: ")
    articles = get_news_articles(query)
    
    # Add try-catch if no articles found (error processing)
    if not articles:
        print("No articles found for the query.")
        return

    # Randomly pick 5 articles from the fetched articles
    sampled_articles = random.sample(articles, min(5, len(articles)))
    
    results = []
    for article in sampled_articles:
        content = article['description'] or article['content']
        analysis_result = analyze_relationships(content)
        if analysis_result:
            results.append(analysis_result)
    
    # Parse the results and create a DataFrame
    parsed_results = []
    for result in results:
        lines = result.split("\n")
        for line in lines:
            if line:
                parts = line.split(',')
                if len(parts) == 3:
                    parsed_results.append(parts)
    
    df = pd.DataFrame(parsed_results, columns=['Company1', 'Company2', 'RelationshipType'])
    df.to_csv('company_relationships.csv', index=False)
    print("Results saved to 'company_relationships.csv'.")

if __name__ == "__main__":
    main()
