# Dynamic Batching System

This module provides a dynamic batching system that automatically adjusts batch sizes based on hardware capabilities and performance metrics.

## Features

- Automatic batch size adjustment based on:
  - Processing time
  - Memory usage
  - Input data shapes
  - Hardware capabilities
- Performance monitoring and metric collection
- Auto-tuning capabilities
- Batch processing decorator for easy integration
- Integration with PyTorch DataLoader
- Optimization for different input shapes

## Components

### DynamicBatcher

The main class that handles batch size adjustments and performance monitoring.

```python
batcher = DynamicBatcher(
    initial_batch_size=16,
    min_batch_size=1,
    max_batch_size=64,
    target_memory_usage=0.7,
    target_latency=0.5,
    growth_factor=1.2,
    reduction_factor=0.8,
    auto_tune=True,
    enable_monitoring=True
)
```

### BatchMetrics

A data class that stores metrics about batch processing.

### batch_processor

A decorator that wraps any processing function to handle dynamic batching automatically.

## Usage Examples

### Basic Usage

```python
# Initialize a dynamic batcher
batcher = DynamicBatcher(
    initial_batch_size=16,
    min_batch_size=1,
    max_batch_size=64,
    target_memory_usage=0.7
)

# Process data in batches
for i in range(0, len(data), batcher.get_batch_size()):
    batch = data[i:i+batcher.get_batch_size()]
    
    # Process batch
    start_time = time.time()
    result = process_batch(batch)
    processing_time = time.time() - start_time
    
    # Update batch size for next iteration
    batcher.adjust_batch_size(
        processing_time=processing_time,
        batch_shape=batch.shape
    )
```

### Using the Decorator

```python
# Initialize a dynamic batcher
batcher = DynamicBatcher(initial_batch_size=16)

# Define a function that processes a single batch
@batch_processor(batcher)
def process_batch(batch):
    # Process the batch
    return [process_item(item) for item in batch]

# Process all data automatically with the decorator
results = process_batch(all_data)
```

### Integration with PyTorch DataLoader

```python
# Initialize batcher
batcher = DynamicBatcher(initial_batch_size=32, auto_tune=True)

# Create DataLoader with batch size from batcher
dataloader = DataLoader(
    dataset,
    batch_size=batcher.get_batch_size(),
    num_workers=batcher.get_optimal_workers(),
    shuffle=True
)

# Process batches
for images, labels in dataloader:
    # Process batch
    start_time = time.time()
    outputs = model(images)
    processing_time = time.time() - start_time
    
    # Update batch size
    new_batch_size = batcher.adjust_batch_size(
        processing_time=processing_time,
        batch_shape=images.shape
    )
    
    # Recreate dataloader if batch size changed
    if new_batch_size != dataloader.batch_size:
        dataloader = DataLoader(
            dataset,
            batch_size=new_batch_size,
            num_workers=batcher.get_optimal_workers(),
            shuffle=True
        )
```

### Optimizing for Input Shapes

```python
# Get optimized batch size for a specific input shape
optimal_size = batcher.optimize_for_input_size((3, 224,, 224))
```

## Metrics and Monitoring

The DynamicBatcher collects performance metrics during processing:

```python
# Get metrics
metrics = batcher.get_metrics()

# Save metrics to file
metrics_file = batcher.save_metrics()
```

## See Also

For detailed examples, see `example.py`. 