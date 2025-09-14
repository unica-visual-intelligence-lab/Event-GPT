from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config
import numpy as np
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

EMBED_DIMS = {
    'OpenGVLab/VideoMAEv2-Base' : 768,  # VideoMAE Base model
    'openai-community/gpt2': 768,  # GPT-2 Base model
}
VIDEO_MODEL_NAME = 'OpenGVLab/VideoMAEv2-Base'
TEXT_MODEL_NAME = 'openai-community/gpt2'

def get_video_encoder():
    """
    Returns a VideoMAE model for video encoding.
    """
    config = AutoConfig.from_pretrained(VIDEO_MODEL_NAME, trust_remote_code=True)
    processor = VideoMAEImageProcessor.from_pretrained(VIDEO_MODEL_NAME)
    model = AutoModel.from_pretrained(VIDEO_MODEL_NAME, config=config, trust_remote_code=True)
    # Freeze the model parameters
    return processor, model


def get_autoregressive_model(num_labels=10):
    """
    Returns a GPT-2 model for autoregressive text generation.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = GPT2ForSequenceClassification.from_pretrained(TEXT_MODEL_NAME, 
                                                         num_labels=num_labels,
                                                         problem_type="multi_label_classification",
                                                         trust_remote_code=True)
    # LoRA configuration
    lora_config = LoraConfig(
        r=8,                          # Rank
        lora_alpha=16,
        target_modules=["c_attn"],   # GPT2 attention input layer
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS   # Since you're doing classification
    )

    model = get_peft_model(model, lora_config)

    return tokenizer, model




class EventClassifier(nn.Module):
    """
    A simple event classifier that uses a video encoder and an autoregressive model.
    """
    def __init__(self, num_labels=10):
        super(EventClassifier, self).__init__()
        self.processor, self.video_encoder = get_video_encoder()
        _, self.text_model = get_autoregressive_model(num_labels)
        DEBUG= False
        if DEBUG:
            print("Video Encoder:", self.video_encoder)
            print("Text Model:", self.text_model)
        #linear layer to pass from video encoder to text model
        self.linear = nn.Linear(
            in_features=EMBED_DIMS[VIDEO_MODEL_NAME],
            out_features=EMBED_DIMS[TEXT_MODEL_NAME]
        )
        #classifier head
        self.classifier = nn.Linear(
            in_features=EMBED_DIMS[TEXT_MODEL_NAME],
            out_features=num_labels
        )
        # Plug in the classifier head to the text model
        #self.text_model.classifier = self.classifier
        #CONTROLLERS
        self.controller={
            'video_encoder_frozen': True,
            'text_model_frozen': True,
            'linear_frozen': False
        }
        # Freeze the video encoder and text model if specified
        if self.controller['video_encoder_frozen']:
            for param in self.video_encoder.parameters():
                param.requires_grad = False
        
        if self.controller['linear_frozen']:
            for param in self.linear.parameters():
                param.requires_grad = False

        print("Text Model total parameters:", sum(p.numel() for p in self.text_model.parameters()), "trainable parameters:", sum(p.numel() for p in self.text_model.parameters() if p.requires_grad))
        print("Video Encoder total parameters:", sum(p.numel() for p in self.video_encoder.parameters()), "trainable parameters:", sum(p.numel() for p in self.video_encoder.parameters() if p.requires_grad))

    def forward(self, video_inputs,old_past_key_values=None,labels=None):
        #video_inputs have shape (16, 3, 224, 224) and type np.ndarray
        # B, T, C, H, W -> B, C, T, H, W
        inputs = self.processor(video_inputs, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
        # Perform inference with the video encoder
        #to the device
        inputs = {k: v.to(self.video_encoder.device) for k, v in inputs.items()}
        video_embeddings = self.video_encoder(**inputs)
        # now with shape (1,768)
        # Pass the video embeddings through the linear layer
        video_embeddings = self.linear(video_embeddings)
        #epad the video embeddings to match the text model's input shape
        video_embeddings = video_embeddings.unsqueeze(1)  # Add a sequence dimension
        #fed to the text model
        # If old_past_key_values is provided, use it to continue the sequence
        if old_past_key_values is not None:
            video_embeddings = torch.cat((old_past_key_values, video_embeddings), dim=1)
        if labels is  None:

            prediction = self.text_model(
                    inputs_embeds=video_embeddings,
                    #past_key_values=old_past_key_values,
                    output_hidden_states =True,
                    return_dict=True
            )
        
        
            return prediction.logits, prediction.hidden_states[-1]  # Return logits and last hidden state as past_key_values
        else:
            # If labels are provided, compute the loss
            prediction = self.text_model(
                inputs_embeds=video_embeddings,
                labels=labels,
                #past_key_values=old_past_key_values,
                output_hidden_states=True,
                return_dict=True
            )
            # Return logits and loss
            return prediction.logits, prediction.hidden_states[-1], prediction.loss

if __name__ == "__main__":
    # Example usage
    model = EventClassifier(num_labels=10)
    print(model)
    #print(model)
    # Example video input
    video = list(np.random.rand(16, 3, 224, 224))
    # predict the video embeddings
    with torch.no_grad():
        prediction, past_emb = model(video)
    # Print the shape of the prediction
    print("Prediction shape:", prediction.shape)
    # Print the prediction
    print("Prediction =", prediction)
    print("Prediction =", prediction.argmax().item())
    #use the past_key_values to predict the next frame
    next_frame = list(np.random.rand(16, 3, 224, 224))
    with torch.no_grad():
        next_prediction, _ = model(next_frame, old_past_key_values=past_emb)
    # Print the shape of the next prediction
    print("Next Prediction shape:", next_prediction.shape)
    # Print the next prediction
    print("Next Prediction =", next_prediction)
    # Print the next prediction
    print("Next Prediction =", next_prediction.argmax().item())


















'''Example usage of the video encoder
video = list(np.random.rand(16, 3, 224, 224))
# B, T, C, H, W -> B, C, T, H, W
inputs = processor(video, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
with torch.no_grad():
  outputs = model(**inputs)
print(outputs.shape)
'''
'''Example usage of the autoregressive model
# Example text input
text = "This is an example text input for classification."
# Tokenize the text input
inputs_text = tokenizer(text, return_tensors="pt")
#print the shape of the inputs_text
print(inputs_text['input_ids'].shape)
# Perform inference
with torch.no_grad():
    text_outputs = text_model(**inputs_text)
# Get the logits for classification
logits = text_outputs.logits
# Print the logits
print(logits)
'''


