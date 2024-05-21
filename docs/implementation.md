# Notes on implementation

- I used UniProt to get a dataset with 10Mio. enzymes just by searching for 'enzyme'
- As export format I used a [`.fasta` format](https://zhanggroup.org/FASTA/) 
- I imported the file into my notebook using Biopython
- I put my dataset into S3 R2 Cloudflare and sync with rclone