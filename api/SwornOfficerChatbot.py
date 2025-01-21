from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_debug.log'),
        logging.StreamHandler()
    ]
)

print("All environment variables:")
for key, value in os.environ.items():
    if "OPENAI_API_KEY" in key or "PINECONE_API_KEY" in key:
        print(f"{key}: {value}")

# Load environment variables
load_dotenv("/Users/jameschan/PycharmProjects/content-chatbot-sworn/api/.env", override=True)

print("openai api key: ", os.getenv('OPENAI_API_KEY'))

# Initialize clients
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("swornvideorecommendationsystem")

if not os.getenv('OPENAI_API_KEY') or not os.getenv('PINECONE_API_KEY'):
    raise ValueError("Missing required environment variables. Please check OPENAI_API_KEY and PINECONE_API_KEY")


class ContentChatbot:
    def __init__(self):
        self.conversation_history = []
        self.logger = logging.getLogger(__name__)

    def generate_embedding(self, text):
        """Generate embedding for search query"""
        self.logger.info(f"Generating embedding for text: {text[:100]}...")
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            self.logger.debug(f"Successfully generated embedding of length {len(response.data[0].embedding)}")
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise

    def search_content(self, query, top_k=3):
        """Search both namespaces for relevant content"""
        self.logger.info(f"Searching content for query: {query}")
        query_embedding = self.generate_embedding(query)

        results = {'ns1': None, 'ns2': None}

        try:
            # Search video transcripts (ns1)
            self.logger.debug("Searching namespace ns1 (video transcripts)...")
            video_results = index.query(
                namespace="ns1",
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            self.logger.info(
                f"Found {len(video_results.matches) if hasattr(video_results, 'matches') else 0} results in ns1")
            results['ns1'] = video_results

            # Log detailed results for ns1
            if hasattr(video_results, 'matches'):
                for i, match in enumerate(video_results.matches):
                    self.logger.debug(f"NS1 Match {i + 1}:")
                    self.logger.debug(f"Score: {match.score}")
                    self.logger.debug(f"Metadata: {json.dumps(match.metadata, indent=2)}")
        except Exception as e:
            self.logger.error(f"Error searching ns1: {str(e)}")
            results['ns1'] = None

        try:
            # Search other content (ns2)
            self.logger.debug("Searching namespace ns2 (educational content)...")
            content_results = index.query(
                namespace="ns2",
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            self.logger.info(
                f"Found {len(content_results.matches) if hasattr(content_results, 'matches') else 0} results in ns2")
            results['ns2'] = content_results

            # Log detailed results for ns2
            if hasattr(content_results, 'matches'):
                for i, match in enumerate(content_results.matches):
                    self.logger.debug(f"NS2 Match {i + 1}:")
                    self.logger.debug(f"Score: {match.score}")
                    self.logger.debug(f"Metadata: {json.dumps(match.metadata, indent=2)}")
        except Exception as e:
            self.logger.error(f"Error searching ns2: {str(e)}")
            results['ns2'] = None

        return results['ns1'], results['ns2']

    def safe_get_metadata(self, match):
        """Safely extract metadata from a match"""
        self.logger.debug(f"Extracting metadata from match: {match}")
        try:
            source = match.metadata.get('text-source', 'Unknown source')
            content = match.metadata.get('content', 'No content available')
            self.logger.debug(f"Successfully extracted metadata - Source: {source}")
            return source, content
        except AttributeError as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            return 'Unknown source', 'No content available'

    def format_context(self, video_results, content_results):
        """Format search results into context for the LLM"""
        self.logger.info("Formatting context from search results")
        context = []

        # Add content from ns2 (educational materials)
        if content_results and hasattr(content_results, 'matches'):
            self.logger.debug(f"Processing {len(content_results.matches)} matches from ns2")
            context.append("Relevant information from educational content:")
            for i, match in enumerate(content_results.matches):
                source, content = self.safe_get_metadata(match)
                self.logger.debug(f"NS2 Content {i + 1} from {source}: {content[:100]}...")
                context.append(f"\nFrom {source}:\n{content}")
        else:
            self.logger.debug("No results from ns2 or invalid format")

        # Add content from ns1 (video transcripts)
        if video_results and hasattr(video_results, 'matches'):
            self.logger.debug(f"Processing {len(video_results.matches)} matches from ns1")
            context.append("\nRelevant information from video transcripts:")
            for i, match in enumerate(video_results.matches):
                source, content = self.safe_get_metadata(match)
                self.logger.debug(f"NS1 Content {i + 1} from {source}: {content[:100]}...")
                context.append(f"\nFrom video {source}:\n{content}")
        else:
            self.logger.debug("No results from ns1 or invalid format")

        if not context:
            self.logger.warning("No relevant content found in either namespace")
            context.append("No relevant content found in the knowledge base.")

        formatted_context = "\n".join(context)
        self.logger.info(f"Final formatted context length: {len(formatted_context)} characters")
        self.logger.debug("Formatted Context:")
        self.logger.debug("=" * 50)
        self.logger.debug(formatted_context)
        self.logger.debug("=" * 50)
        return formatted_context

    def generate_response(self, user_query, context):
        """Generate a response using the OpenAI API"""
        self.logger.info("Generating response from OpenAI")
        try:
            messages = [
                {"role": "system", "content": """You are a helpful assistant with access to a knowledge base of educational
                 content and video transcript, curated specifically to help on duty police officers manage their health.
                 Use the provided context to answer questions accurately. Be conversational but precise.
                 When you reference information, mention if it comes from educational content or video material.
                 If the context doesn't contain relevant information, be honest about it and provide general guidance."""},
                {"role": "user", "content": f"""Context:\n{context}\n\nUser Question: {user_query}
                 Please provide a helpful response using this context. If the context doesn't contain relevant information,
                 let me know and provide general guidance instead."""}
            ]

            messages.extend(self.conversation_history)

            self.logger.debug(f"Sending request to OpenAI with {len(messages)} messages")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            self.logger.info("Successfully received response from OpenAI")
            self.logger.debug(f"Response content: {response.choices[0].message.content[:100]}...")
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error generating a response. Please try asking your question in a different way."

    def chat(self, user_input):
        """Main chat function that processes user input and returns a response"""
        self.logger.info(f"Processing chat input: {user_input}")
        try:
            self.logger.debug("Starting content search...")
            video_results, content_results = self.search_content(user_input)

            self.logger.debug("Formatting context...")
            context = self.format_context(video_results, content_results)

            self.logger.debug("Generating response...")
            response = self.generate_response(user_input, context)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]

            self.logger.info("Successfully generated and returned response")
            return response

        except Exception as e:
            self.logger.error(f"Error in chat method: {type(e).__name__}, {str(e)}", exc_info=True)
            return f"I apologize, but I encountered an error: {type(e).__name__} - {str(e)}"


def main():
    logging.info("Starting Content Chatbot")
    chatbot = ContentChatbot()

    print("Content Chatbot initialized. Type 'quit' to exit.")
    print("Connecting to knowledge base...")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input.lower() == 'quit':
            logging.info("Shutting down chatbot")
            break

        response = chatbot.chat(user_input)
        print("\nAssistant:", response)


if __name__ == "__main__":
    main()