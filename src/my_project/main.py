"""Entry point for my_project"""
import argparse
from my_project import core

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hello', action='store_true', help='Print hello')
    args = parser.parse_args()
    if args.hello:
        print('Hello from my_project')

if __name__ == '__main__':
    cli()
