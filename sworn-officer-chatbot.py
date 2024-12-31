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


class ContentChatbot:
    def __init__(self):
        self.conversation_history = []

    def generate_embedding(self, text):
        """Generate embedding for search query"""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def search_content(self, query, top_k=3):
        """Search both namespaces for relevant content"""
        query_embedding = self.generate_embedding(query)

        # Search video transcripts (ns1)
        video_results = index.query(
            namespace="ns1",
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Search other content (ns2)
        content_results = index.query(
            namespace="ns2",
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return video_results, content_results

    def format_context(self, video_results, content_results):
        """Format search results into context for the LLM"""
        context = "Relevant information from educational content:\n"

        # Add content from ns2 (educational materials)
        for match in content_results.matches:
            context += f"\nFrom {match.metadata['text-source']}:\n{match.metadata['content']}\n"

        context += "\nRelevant information from video transcripts:\n"

        # Add content from ns1 (video transcripts)
        for match in video_results.matches:
            context += f"\nFrom video {match.metadata['text-source']}:\n{match.metadata['content']}\n"

        return context

    def generate_response(self, user_query, context):
        """Generate a response using the OpenAI API"""
        # Construct the prompt
        messages = [
            {"role": "system", "content": """You are a helpful assistant with access to a knowledge base of educational content and video transcripts. 
             Use the provided context to answer questions accurately. Be conversational but precise.
             When you reference information, mention if it comes from educational content or video material.
             If the context doesn't contain relevant information for the query, be honest about it."""},
            {"role": "user", "content": f"""Context:\n{context}\n\nUser Question: {user_query}
             Please provide a helpful response using this context. If the context doesn't contain relevant information,
             let me know and provide general guidance instead."""}
        ]

        # Add conversation history for context
        messages.extend(self.conversation_history)

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

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
            return f"I apologize, but I encountered an error: {str(e)}"


def main():
    # Example usage
    chatbot = ContentChatbot()

    print("Content Chatbot initialized. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        response = chatbot.chat(user_input)
        print("\nAssistant:", response)


if __name__ == "__main__":
    main()