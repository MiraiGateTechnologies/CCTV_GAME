from multiprocessing import Pool
import subprocess

streams = []
with open("streams.txt") as f:
    for line in f:
        url, name = line.strip().split()
        streams.append((url, name))

def run(args):
    url, name = args
    subprocess.run([
        "python", "download.py",
        "--url", url,
        "--name", name,
        "--duration", "120"
    ])

if __name__ == "__main__":
    with Pool(8) as p:
        p.map(run, streams)