# Notes on implementation

- I used UniProt to get a dataset with 10Mio. enzymes just by searching for 'enzyme'
- As export format I used a [`.fasta` format](https://zhanggroup.org/FASTA/) 
- I imported the file into my notebook using Biopython
- I put my dataset into S3 R2 Cloudflare and sync with rclone
- I setup a container on fly.io like the following: https://fly.io/docs/blueprints/opensshd/
- Issue with openfold: https://github.com/aqlaboratory/openfold/issues/403 -> used the following solution:https://github.com/aqlaboratory/openfold/issues/403#issuecomment-1955260528
- Also incompatabilit of rdkit with pandas >2.2
- +Bipython version conflict (which can be ignored)