import json
import openai
import random
import os
import logging
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
rm_path = "/cosmos/hanyang_critic/RLAIF/data/ultrafeedback/RM"

# Function to load JSON data
def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        return []

# Function to save JSON data
def save_json(data, filepath):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")

# Function to perform critique and annotate spans for general Q&A
def critique_and_annotate(data, sample_size=None):
    # If sample_size is specified, randomly sample the data
    if sample_size is not None:
        data = random.sample(data, min(sample_size, len(data)))
    
    results = []
    for i, entry in enumerate(data):
        logger.info(f"Processing entry {i+1}/{len(data)}")
        # Format the prompt for critique
        formatted_prompt = f"""<CritiquePrompt>
    <Instructions>Critique a response to a user input and provide feedback in JSON format:</Instructions>

    <EvaluationCriteria>
        <Criterion name="Accuracy">Does it correctly address the input?</Criterion>
        <Criterion name="Relevance">Does it stay on topic?</Criterion>
        <Criterion name="Clarity">Is it easy to understand?</Criterion>
        <Criterion name="Completeness">Does it cover the input's core needs?</Criterion>
    </EvaluationCriteria>

    <SpanGuidelines>
        <GoodSpans>
            <Description>Phrases from the response that best capture its strengths (e.g., accurate, relevant, clear). Select only the most essential and impactful phrases, directly tied to 'textual_feedback'.</Description>
        </GoodSpans>
        <PoorSpans>
            <Description>Phrases from the response that best highlight its weaknesses (e.g., inaccurate, irrelevant, vague). Select only the most essential and impactful phrases, directly tied to 'textual_feedback'.</Description>
        </PoorSpans>
        <Requirement>Spans must be exact quotes from the response.</Requirement>
    </SpanGuidelines>

    <ReflectionProcess>
        <Step>First, carefully analyze the user input to understand the core question or request.</Step>
        <Step>Next, examine the generated response against each evaluation criterion.</Step>
        <Step>For each criterion, identify specific strengths and weaknesses with supporting evidence from the response.</Step>
        <Step>Consider how well the response addresses the user's explicit and implicit needs.</Step>
        <Step>Finally, synthesize your analysis into a coherent critique that highlights the most important points.</Step>
    </ReflectionProcess>
 
    <Separator>---</Separator>

    <UserInput>{entry['prompt']}</UserInput>
    <GeneratedResponse>{entry['response']}</GeneratedResponse>

    <Separator>---</Separator>

    <OutputFormat>
        <Description>Provide the critique in the following JSON structure:</Description>
        <JSONExample>
            {{
                "textual_feedback": "One-paragraph critique summarizing strengths and weaknesses, tied to spans.",
                "good_spans": ["phrase1", "phrase2", ...],  // Impactful phrases from <GeneratedResponse>, or [] if none
                "poor_spans": ["phrase1", "phrase2", ...]   // Impactful phrases from <GeneratedResponse>, or [] if none
            }}
        </JSONExample>
    </OutputFormat>
</CritiquePrompt>"""

        logger.info('Finished creating prompt')
        # Get critique and annotations from GPT-4
        try:
            client = OpenAI()
            gpt4_response = client.chat.completions.create(
                model="gpt-4o-20240513",
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.6,
                max_tokens=1200,
                top_p=0.95,
            )
            
            # Parse GPT-4 response
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    gpt4_response_content = gpt4_response.choices[0].message.content.strip()
                    logger.debug(f"GPT-4 response: {gpt4_response_content[:100]}...")
                    critique_data = json.loads(gpt4_response_content)
                    entry['critique'] = critique_data.get('textual_feedback', '')
                    entry['annotations'] = {
                        'good_spans': critique_data.get('good_spans', []),
                        'poor_spans': critique_data.get('poor_spans', [])
                    }
                    break  # Exit loop if parsing is successful
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retrying JSON parse (attempt {attempt + 1}) due to error: {e}")
                    else:
                        logger.error(f"Failed to parse GPT-4 response after {max_retries} attempts: {e}")
                        entry['critique'] = ''
                        entry['annotations'] = {'good_spans': [], 'poor_spans': []}
        except Exception as e:
            logger.error(f"Error getting chat completion: {e}")
            entry['critique'] = ''
            entry['annotations'] = {'good_spans': [], 'poor_spans': []}
        
        results.append(entry)
    
    return results

# Load train and train data
train_data = load_json(f"{rm_path}/train_sep.json")
logger.info('start')

# Function to load progress
def load_progress(progress_file):
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress from {progress_file}: {e}")
    return 0  # Start from the beginning if no progress file exists

# Function to save progress
def save_progress(progress_file, batch_index):
    try:
        with open(progress_file, 'w') as f:
            json.dump(batch_index, f)
        logger.info(f"Progress saved: {batch_index}")
    except Exception as e:
        logger.error(f"Error saving progress to {progress_file}: {e}")

# Process and save all train data in a single file with restore support
def process_and_save_all(data, batch_size, output_path, progress_file=None, start_index=0):
    # Load existing annotated data if it exists
    existing_data = []
    if os.path.exists(output_path):
        existing_data = load_json(output_path)
        logger.info(f"Loaded {len(existing_data)} existing annotated entries from {output_path}")

    # Use progress file if provided
    if progress_file:
        start_index = load_progress(progress_file)
        logger.info(f"Loaded progress: starting from index {start_index}")
    
    # Calculate remaining data to process
    remaining_data = data[start_index:]
    total_batches = (len(remaining_data) + batch_size - 1) // batch_size

    logger.info(f"Starting from index {start_index}, processing {len(remaining_data)} entries")

    for batch_index in range(total_batches):
        batch_start = batch_index * batch_size
        batch_end = min(batch_start + batch_size, len(remaining_data))
        batch_data = remaining_data[batch_start:batch_end]
        
        current_global_index = start_index + batch_end
        logger.info(f"Processing batch {batch_index + 1}/{total_batches} (global index: {current_global_index})")

        # Process current batch
        batch_data_annotated = critique_and_annotate(batch_data)

        # Add batch results to existing data
        existing_data.extend(batch_data_annotated)

        # Save all data after each batch
        save_json(existing_data, output_path)
        
        # Save progress if progress file is provided
        if progress_file:
            save_progress(progress_file, current_global_index)
            
        logger.info(f"Batch {batch_index + 1}/{total_batches} processed. Total entries saved: {len(existing_data)}")

    logger.info(f"All data processed and saved to {output_path}")

# Main execution
if __name__ == "__main__":
    logger.info("Starting annotation process")
    
    # Load train data
    train_data = load_json(f"{rm_path}/train_sep.json")
    logger.info(f"Loaded {len(train_data)} training examples")
    
    # Define progress file
    progress_file = f"{rm_path}/annotation_progress.json"
    
    # Process and save all train data with progress tracking
    process_and_save_all(
        train_data, 
        batch_size=10, 
        output_path=f"{rm_path}/train_annotated_v3.json", 
        progress_file=progress_file,
        start_index=1 
    )
