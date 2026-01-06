"""
Script to run the optimized DataFrame-based ensemble evaluation pipeline.
"""

from pipelines.final_ensemble_pipeline_dataframe import final_ensemble_pipeline_dataframe


if __name__ == "__main__":
    print("Starting optimized DataFrame-based ensemble evaluation pipeline...")
    print("This will evaluate all 16,777,191 possible ensemble combinations.")
    print("The evaluation uses pre-stacking and efficient slicing for better performance.")
    print("\nKey optimizations:")
    print("- Pre-stacks all 36 models into 24 groups")
    print("- Uses DataFrame with pre-populated combination info")
    print("- Slices and averages predictions efficiently")
    print("- Supports checkpointing for resumability")
    print("\nYou can monitor progress in another terminal with:")
    print("python monitor_dataframe_progress.py")
    print("\n" + "="*60 + "\n")
    
    # Run the pipeline
    pipeline_instance = final_ensemble_pipeline_dataframe()
    pipeline_run = pipeline_instance.run()
    
    print("\nPipeline completed successfully!")
    print(f"Run name: {pipeline_run.name}")
    print("\nResults saved to: results/ensemble_checkpoints/evaluation_results.csv")