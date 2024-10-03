# RL Enzyme Engineering

## Run on fly.io

fly.io offers [simple access to GPU nodes](https://fly.io/gpu).

To launch an instance of this algorithm on fly.io follow those steps: 

1. Create an account on [fly.io](fly.io). \
    Beware: to get access to GPU nodes you may need to write them a short mail to ask for access incl. your acounts id, as they are still rolling out their offer.
2. Install the fly.io CLI [flyctl](https://fly.io/docs/flyctl/)
3. Clone the repostiory to your local system:
    ```bash
    git clone git@github.com:li-il-li/rl-enzyme-engineering.git
    ```
4. Create a new 'app' on fly.io:
    ```bash
    cd flyio
    fly launch
    # Would you like to allocate dedicated ipv4 and ipv6 addresses now? y -> copy IP
    ```

5. [Add you SSH public key](https://fly.io/docs/blueprints/opensshd/#upload-your-ssh-key):
    ```bash
    fly secrets set "AUTHORIZED_KEYS=$(cat ~/.ssh/id_rsa.pub)"
    ```
    If you want to work on the machine I recommend also setting:
    ```bash
    fly secrets set "GIT_NAME=<name>"
    fly secrets set "GIT_EMAIL=<email>"
    fly secrets set "GIT_PASSWORD=<password>"
    ```
6. Deploy the app:
    ```bash
    fly deploy
    ```
7. Recommendation: Use VSCode to connect to the instance via SSH or just:
    ```bash
    ssh root@<copied_IP>
    ```
8. Configure your `wildtype_AA_seq` and `ligand_smile` in `conf.yaml`
9. Run algorithm via:
    ```bash
    uv run src/main.py
    ```
6. Important: When you are done, shutdown the instance:
    ```bash
    fly scale count 0
    ```
    This will scale down the instance but keep the volume (costs!). \
    To delete the whole app:
    ```bash
    fly apps destroy rl-enzyme-engineering
    ```
    
