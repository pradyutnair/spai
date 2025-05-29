import pandas as pd
import argparse
import os

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Create combined real and fake image dataset for SPAI evaluation')
    parser.add_argument('--fake-csv', type=str, required=True, help='Path to the fake images CSV file')
    parser.add_argument('--fake-name', type=str, required=True, help='Name of the fake images dataset')
    parser.add_argument('--include-raise', action='store_true', help='Include RAISE dataset in the combined CSV')
    args = parser.parse_args()
    
    # Load real images CSVs
    real_df_imgnet = pd.read_csv('/home/scur2605/spai/data/real_imagenet.csv')
    print(f'ImageNet: Loaded {len(real_df_imgnet)} real images')
    
    real_df_coco = pd.read_csv('/home/scur2605/spai/data/real_coco.csv')
    print(f'COCO: Loaded {len(real_df_coco)} real images')
    
    real_df_fodb = pd.read_csv('/home/scur2605/spai/data/real_fodb.csv')
    print(f'FODB: Loaded {len(real_df_fodb)} real images')
    
    # Sample OpenImages to 1k images
    real_df_open = pd.read_csv('/home/scur2605/spai/data/openimages_test.csv')
    real_df_open = real_df_open.sample(1000, random_state=42)
    print(f'OpenImages: Loaded {len(real_df_open)} real images')
    
    # Load RAISE dataset if requested
    if args.include_raise:
        real_df_raise = pd.read_csv('/home/scur2605/spai/data/real_raise.csv')
        print(f'RAISE: Loaded {len(real_df_raise)} real images')
    
    # Load fake/AI-generated images CSV from command-line argument
    fake_df = pd.read_csv(args.fake_csv)
    print(f"Loaded {len(fake_df)} fake images from {args.fake_csv}")
    
    # Concatenate the dataframes
    if args.include_raise:
        df = pd.concat([real_df_imgnet, real_df_coco, real_df_fodb, real_df_raise, real_df_open, fake_df], ignore_index=True)
    else:
        df = pd.concat([real_df_imgnet, real_df_coco, real_df_fodb, real_df_open, fake_df], ignore_index=True)
    
    # Save the concatenated dataframe to a new CSV file
    output = f'/home/scur2605/spai/data/combined_real_fake_{args.fake_name}.csv'
    df.to_csv(output, index=False)
    print(f"Combined dataframe saved to {output} with {len(df)} images")

if __name__ == "__main__":
    main()