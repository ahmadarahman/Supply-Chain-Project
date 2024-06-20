import openai
import requests
import gnews

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
        
gptKey = get_file_contents('keys/gpt_key.txt')
gnewsKey = get_file_contents('keys/gnews_key.txt')

# Passing in API keys for OpenAI and gnews
openai.api_key = gptKey
gnews_api_key = gnewsKey
# Function to fetch article content from a specific URL
def fetch_article_content(url):
    # Replace this with actual logic to fetch article content from the URL
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch article from {url}. Status code: {response.status_code}")
        return None

# Function to classify relationships between companies using OpenAI's Chat Completions API
def classify_relationships(article_content):
    prompt = f"Can you process the following article?\n\n{article_content}\n\nWhat are the companies mentioned in the article? " \
             "For every company pair, assign their relationship to one of the following types:\n\n" \
             "1. Acquisition\n" \
             "2. Strategic collaboration\n" \
             "3. Competitor relationship\n" \
             "4. Supplier and subsidiary relationship\n" \
             "5. Customer client relationship\n" \
             "6. Regulatory and government body relationship\n\n"

    # Call OpenAI's Chat Completions API to generate the relationships
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-instruct",  # Specify the GPT-4 model you want to use
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": ""}
        ],
        max_tokens=150,  # Adjust the number of tokens as needed
        stop=["\n\n"]  # Stop generation at double newline to get structured response
    )

    # Process the response to extract relationships
    relationships = []
    for message in response['messages']:
        if message['role'] == 'system':
            continue
        text = message['content'].strip()
        if text:
            relationships.append(text.split(','))

    return relationships

# Function to save relationships to a CSV file
def save_to_csv(relationships, filename='company_relationships.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Company 1', 'Company 2', 'Relationship Type'])
        for relationship in relationships:
            writer.writerow([relationship[0].strip(), relationship[1].strip(), relationship[2].strip()])
    print(f"Relationships saved to {filename}")

# Main function to process a specific article URL
def main():
    article_url = input("Enter the URL of the news article you want to process: ")
    article_content = fetch_article_content(article_url)
    if article_content:
        relationships = classify_relationships(article_content)
        save_to_csv(relationships)

# Run the main function
if __name__ == "__main__":
    main()
