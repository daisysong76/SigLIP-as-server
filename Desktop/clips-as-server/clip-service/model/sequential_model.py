    def forward(self, inputs):
        """Forward pass through the model."""
        # Handle empty batches
        if len(inputs["input_ids"]) == 0:
            # Return empty tensor with appropriate shape
            return {"embeddings": torch.zeros((0, self.embedding_dim), device=self.device)}
        
        # Process inputs through CLIP
        # We may need to handle padding if batches have different lengths
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"] 
        attention_mask = inputs.get("attention_mask", None)
        
        # Create proper attention mask if not provided
        if attention_mask is None and "attention_mask" not in inputs:
            attention_mask = torch.ones_like(input_ids)
            
        # Forward through model
        outputs = self.clip_model(
            input_ids=input_ids, 
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=True
        ) 