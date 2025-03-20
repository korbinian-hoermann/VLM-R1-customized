import os
import pandas as pd
import wandb
from datetime import datetime
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Union

class TrainingTracker:
    """
    A class to track VLM agent training, including prompts, images, annotations,
    evaluation responses, and metrics.
    """
    
    def __init__(self, log_to_wandb: bool = True, log_dir: Optional[str] = None):
        """
        Initialize the tracker.
        
        Args:
            log_to_wandb: Whether to log to Weights & Biases
            log_dir: Directory to save local logs (if None, will use current time)
        """
        self.log_to_wandb = log_to_wandb
        self.log_dir = log_dir or f"tracking_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize the tracking dataframe
        self.tracking_df = pd.DataFrame(columns=[
            'sample_id',
            'timestamp',
            'prompt',
            'image_path',
            'image',
            'annotated_image',
            'model_response',
            'ground_truth',
            'low_level_action_evaluation_reasoning',
            'low_level_action_evaluation_score',
            'high_level_action_evaluation_reasoning',
            'high_level_action_evaluation_score',
            'custom_format_reward_score',
        ])
        
        self.batch_records = []
        self.current_batch = 0
        self.wandb_table = None
        
        # Initialize a single W&B table if logging to W&B

        print("Init of tracking table")
        print(self.batch_records)
        print(self.log_to_wandb)
        print(self.wandb_table)
        print(wandb.run)

        if self.log_to_wandb and wandb.run is not None and self.wandb_table is None:
            print("Initiated W&B table")
            self.wandb_table = wandb.Table(columns=list(self.tracking_df.columns))
    
    def add_sample(self, 
                  sample_id: str,
                  prompt: str,
                  image_path: Optional[str] = None,
                  image: Optional[Image.Image] = None,
                  annotated_image: Optional[Image.Image] = None,
                  model_response: Optional[str] = None,
                  ground_truth: Optional[str] = None,
                  low_level_action_evaluation_reasoning: Optional[str] = None,
                  low_level_action_evaluation_score: Optional[float] = None,
                  high_level_action_evaluation_reasoning: Optional[str] = None,
                  high_level_action_evaluation_score: Optional[float] = None,
                  custom_format_reward_score: Optional[float] = None,
                  ):
        """
        Add a sample to the tracking dataframe.
        
        Args:
            sample_id: Unique identifier for the sample
            prompt: The prompt given to the model
            image_path: Path to the image file
            image: PIL Image object of the input image
            annotated_image: PIL Image object with annotations
            model_response: The model's response
            ground_truth: The ground truth answer
            evaluation_response: Response from the evaluation API
            reward_score: Numerical reward/score
            reward_type: Type of reward (e.g., 'accuracy', 'format')
        """
        record = {
            'sample_id': sample_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prompt': prompt,
            'image_path': image_path,
            'image': self._convert_image_to_base64(image) if image else None,
            'annotated_image': self._convert_image_to_base64(annotated_image) if annotated_image else None,
            'model_response': model_response,
            'ground_truth': ground_truth,
            'low_level_action_evaluation_reasoning': low_level_action_evaluation_reasoning,
            'low_level_action_evaluation_score': low_level_action_evaluation_score,
            'high_level_action_evaluation_reasoning': high_level_action_evaluation_reasoning,
            'high_level_action_evaluation_score': high_level_action_evaluation_score,
            'custom_format_reward_score': custom_format_reward_score,
        }
        
        self.batch_records.append(record)
    
    def _convert_image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL image to base64 string for storage in dataframe"""
        if img is None:
            return None
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string back to PIL Image"""
        if base64_str is None:
            return None
        
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))
    
    def update_tracking_table(self):
        """
        Update the tracking dataframe with the current batch of records
        and log to Weights & Biases if enabled. Every 10 batches, a new table
        is logged to update the GUI visualization.
        """
        if not self.batch_records:
            return

        print("Updateing tracking table")
        print(self.batch_records)
        print(self.log_to_wandb)
        print(self.wandb_table)
        print(wandb.run)

        # Initialize a single W&B table if logging to W&B and not already created
        if self.log_to_wandb and wandb.run is not None and self.wandb_table is None:
            print("Initiated W&B table")
            self.wandb_table = wandb.Table(columns=list(self.tracking_df.columns))
        else:
            print("Using existing W&B table")
        
        # Add batch records to dataframe
        batch_df = pd.DataFrame(self.batch_records)
        self.tracking_df = pd.concat([self.tracking_df, batch_df], ignore_index=True)
        print("Updated tracking table")
        print(self.tracking_df.shape)
        
        # Log the new batch to the current W&B table if enabled
        if self.log_to_wandb and wandb.run is not None:
            if self.wandb_table is None:
                self.wandb_table = wandb.Table(columns=list(self.tracking_df.columns))
            columns = list(self.tracking_df.columns)
            
            # Add rows to the W&B table
            for n, row in batch_df.iterrows():
                print(f"Adding row {n} of new batch_df to W&B table")
                # Handle images specially for W&B
                wandb_row = list(row)
                # Process images for W&B (convert base64 back to image objects)
                image_idx = columns.index('image')
                print(f"Image index: {image_idx}")
                annotated_idx = columns.index('annotated_image')
                print(f"Annotated image index: {annotated_idx}")
                
                if row['image'] is not None:
                    print(f"Adding image to W&B table")
                    img = self._base64_to_image(row['image'])
                    wandb_row[image_idx] = wandb.Image(img)
                    print(f"Added image to W&B table")
                else:
                    wandb_row[image_idx] = None
                    
                if row['annotated_image'] is not None:
                    print(f"Adding annotated image to W&B table")
                    ann_img = self._base64_to_image(row['annotated_image'])
                    wandb_row[annotated_idx] = wandb.Image(ann_img)
                else:
                    wandb_row[annotated_idx] = None
                
                # Add to the existing table
                self.wandb_table.add_data(*wandb_row)

            # Log the current table snapshot under a general key and a batch-specific key
            wandb.log({"training_samples": self.wandb_table})

            # log table with batch number
            wandb.log({f"training_samples_batch_{self.current_batch}": self.wandb_table})
        
        # Clear the current batch records and increment the batch counter
        self.batch_records = []
        self.current_batch += 1

        # Every 10 batches, create a new cumulative table and log it with a new key.
        if self.current_batch % 10 == 0:
            if self.log_to_wandb and wandb.run is not None:
                new_table = wandb.Table(columns=list(self.tracking_df.columns))
                for _, row in self.tracking_df.iterrows():
                    row_list = list(row)
                    image_idx = list(self.tracking_df.columns).index('image')
                    annotated_idx = list(self.tracking_df.columns).index('annotated_image')
                    if row['image'] is not None:
                        row_list[image_idx] = wandb.Image(self._base64_to_image(row['image']))
                    else:
                        row_list[image_idx] = None
                    if row['annotated_image'] is not None:
                        row_list[annotated_idx] = wandb.Image(self._base64_to_image(row['annotated_image']))
                    else:
                        row_list[annotated_idx] = None
                    new_table.add_data(*row_list)
                # Log the new cumulative table under a unique key to refresh the GUI visualization
                wandb.log({f"training_samples_cumulative_{self.current_batch}": new_table})

