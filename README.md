# Team Assignment Optimizer

An optimization tool that assigns students to teams and roles based on their preferences and other constraints using Google OR-Tools.

## Features

- Assigns students to teams while considering:
  - Role preferences (1-2 roles per student)
  - Grade level affinity
  - Gender grouping requirements
- Uses constraint programming to find optimal solutions
- Configurable weights for different optimization criteria

## Requirements

- Python 3.8+
- See `requirements.txt` for Python package dependencies

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/team-optimizer.git
cd team-optimizer
```

2. Create and activate virtual environment:
```bash
make setup
source venv/bin/activate
```

## Usage

1. Prepare your input data in CSV format with the following columns:
   - ID
   - Name (first, last)
   - Gender
   - Grade
   - Role preference scores for each role

2. Run the optimizer:
```bash
make run
```

Or run directly with Python:
```bash
python team_optimizer.py
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── Makefile
├── team_optimizer.py
├── tests/
│   └── test_optimizer.py
└── example_data.csv
```

## Make Commands

- `make setup`: Create virtual environment and install dependencies
- `make clean`: Remove virtual environment and cache files
- `make test`: Run unit tests
- `make format`: Format code using Black
- `make lint`: Run Flake8 linter
- `make run`: Run the optimizer

## Testing

Run the test suite:
```bash
make test
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request