import { Pinecone } from "@pinecone-database/pinecone";
import { FeatureExtractionPipeline, pipeline } from "@xenova/transformers";
import { modelname, namespace, topK } from "./app/config";
import { HfInference } from '@huggingface/inference'

const hf = new HfInference(process.env.HF_TOKEN)
export async function queryPineconeVectorStore(
  client: Pinecone,
  indexName: string,
  namespace: string,
  query: string
): Promise<string> {
  try {
    const apiOutput = await hf.featureExtraction({
      model: "mixedbread-ai/mxbai-embed-large-v1",
      inputs: query,
    });
    console.log(apiOutput);
    
    if (!apiOutput || !Array.isArray(apiOutput)) {
      console.error("Invalid embedding output:", apiOutput);
      return "Error: Failed to generate embeddings";
    }
    
    const queryEmbedding = Array.from(apiOutput);
    const index = client.Index(indexName);
    const queryResponse = await index.namespace(namespace).query({
      topK: 5,
      vector: queryEmbedding as any,
      includeMetadata: true,
      includeValues: false
    });

    console.log(queryResponse);
    
    if (queryResponse.matches.length > 0) {
      const concatenatedRetrievals = queryResponse.matches
        .map((match, index) => {
          if (match.metadata?.chunk) {
            return `\nLegal Finding ${index+1}: \n ${match.metadata.chunk}`;
          }
          return `\nLegal Finding ${index+1}: \n No content available`;
        })
        .join(". \n\n");
      return concatenatedRetrievals;
    } else {
      return "No relevant legal findings were found for your query.";
    }
  } catch (error) {
    console.error("Error in queryPineconeVectorStore:", error);
    return "Error occurred while retrieving relevant information. Please try again.";
  }
}
