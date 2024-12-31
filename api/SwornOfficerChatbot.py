from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("swornvideorecommendationsystem")


if not os.getenv('OPENAI_API_KEY') or not os.getenv('PINECONE_API_KEY'):
    raise ValueError("Missing required environment variables. Please check OPENAI_API_KEY and PINECONE_API_KEY")


class ContentChatbot:
    def __init__(self):
        self.conversation_history = []

    def generate_embedding(self, text):
        """Generate embedding for search query"""
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def search_content(self, query, top_k=3):
        """Search both namespaces for relevant content"""
        query_embedding = self.generate_embedding(query)

        try:
            # Search video transcripts (ns1)
            video_results = index.query(
                namespace="ns1",
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            print(f"Warning: Error searching ns1: {str(e)}")
            video_results = None

        try:
            # Search other content (ns2)
            content_results = index.query(
                namespace="ns2",
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            print(f"Warning: Error searching ns2: {str(e)}")
            content_results = None

        return video_results, content_results

    def safe_get_metadata(self, match):
        """Safely extract metadata from a match"""
        try:
            source = match.metadata.get('text-source', 'Unknown source')
            content = match.metadata.get('content', 'No content available')
            return source, content
        except AttributeError:
            return 'Unknown source', 'No content available'

    def format_context(self, video_results, content_results):
        """Format search results into context for the LLM"""
        context = []

        # Add content from ns2 (educational materials)
        if content_results and hasattr(content_results, 'matches'):
            context.append("Relevant information from educational content:")
            for match in content_results.matches:
                source, content = self.safe_get_metadata(match)
                context.append(f"\nFrom {source}:\n{content}")

        # Add content from ns1 (video transcripts)
        if video_results and hasattr(video_results, 'matches'):
            context.append("\nRelevant information from video transcripts:")
            for match in video_results.matches:
                source, content = self.safe_get_metadata(match)
                context.append(f"\nFrom video {source}:\n{content}")

        if not context:
            context.append("No relevant content found in the knowledge base.")

        return "\n".join(context)

    def generate_response(self, user_query, context):
        """Generate a response using the OpenAI API"""
        try:
            # Construct the prompt
            messages = [
                {"role": "system", "content": """You are a helpful assistant with access to a knowledge base of educational content and video transcripts. 
                 Use the provided context to answer questions accurately. Be conversational but precise.
                 When you reference information, mention if it comes from educational content or video material.
                 If the context doesn't contain relevant information, be honest about it and provide general guidance."""},
                {"role": "user", "content": f"""Context:\n{context}\n\nUser Question: {user_query}
                 Please provide a helpful response using this context. If the context doesn't contain relevant information,
                 let me know and provide general guidance instead."""}
            ]

            # Add conversation history for context
            messages.extend(self.conversation_history)

            # Generate response
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"I apologize, but I encountered an error generating a response. Please try asking your question in a different way."

    def chat(self, user_input):
        """Main chat function that processes user input and returns a response"""
        try:
            # Search for relevant content
            video_results, content_results = self.search_content(user_input)

            # Format the context from search results
            context = self.format_context(video_results, content_results)

            # Generate response
            response = self.generate_response(user_input, context)

            # Update conversation history (keep last 6 messages)
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]

            return response

        except Exception as e:
            return "I apologize, but I encountered an error. Please try asking your question again."


def main():
    # Example usage
    chatbot = ContentChatbot()

    print("Content Chatbot initialized. Type 'quit' to exit.")
    print("Connecting to knowledge base...")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input.lower() == 'quit':
            break

        response = chatbot.chat(user_input)
        print("\nAssistant:", response)


if __name__ == "__main__":
    main()
