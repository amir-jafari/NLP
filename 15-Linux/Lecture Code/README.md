# Linux Shell Scripting for NLP - Learning Guide

This directory contains a comprehensive collection of shell scripts and supporting files designed to teach Linux command-line skills specifically for Natural Language Processing (NLP) applications. The materials progress from basic shell scripting concepts to advanced NLP workflows.

## 📁 Directory Structure

```
15-Linux/
├── Lecture Code/           # Main learning materials
│   ├── Shell Scripts (01-10)
│   ├── Python support files
│   └── Sample data files
└── Class Ex/              # Practice exercises
    ├── Class_Ex_Linux.txt
    └── Class_Ex_Linux_sol.txt
```

## 🎯 Learning Objectives

By working through these materials, students will learn:
- Basic shell scripting syntax and control structures
- Advanced text processing with Linux command-line tools
- Performance optimization and parallel processing
- NLP-specific workflows and pipelines
- Integration of shell scripts with Python programs

## 📚 Script Overview and Learning Path

### **Phase 1: Shell Scripting Fundamentals**

#### 1. `00-basic_command.py` - Linux Command Primer
**Purpose**: Introduction to essential Linux commands through Python
**Key Concepts**:
- File operations (ls, cat, head, tail, touch, rm, cp, mv)
- Process management (ps, kill)
- File permissions (chmod)
- Text searching (grep, find, locate)
- System information (date, uptime, uname)
- Compression (tar, gzip)
- Network commands (ping, wget)

**How to Run**:
```bash
cd "15-Linux/Lecture Code"
python 00-basic_command.py
```

#### 2. `01-basic_if.sh` - Conditional Statements
**Purpose**: Learn basic if-then-fi syntax
**Key Concepts**:
- Variable assignment and comparison
- Conditional execution
- Basic shell script structure

**How to Run**:
```bash
chmod +x 01-basic_if.sh
./01-basic_if.sh
```

#### 3. `02-basic_for.sh` - Loop Structures
**Purpose**: Introduction to for loops
**Key Concepts**:
- Iterating over lists
- Variable expansion in loops
- Basic loop syntax

**How to Run**:
```bash
chmod +x 02-basic_for.sh
./02-basic_for.sh
```

#### 4. `03-nested_if.sh` - Complex Conditionals
**Purpose**: Advanced conditional logic
**Key Concepts**:
- if-elif-else structures
- Numeric comparisons (-eq, -gt, -lt)
- Multi-condition logic

**How to Run**:
```bash
chmod +x 03-nested_if.sh
./03-nested_if.sh
```

#### 5. `04-loop_if.sh` - Combining Control Structures
**Purpose**: Integration of loops and conditionals
**Key Concepts**:
- Nested control structures
- Conditional logic within loops
- Complex program flow

**How to Run**:
```bash
chmod +x 04-loop_if.sh
./04-loop_if.sh
```

### **Phase 2: Performance and Advanced Techniques**

#### 6. `05-speed_demo.sh` - Performance Benchmarking
**Purpose**: Compare different approaches to text processing for performance
**Key Concepts**:
- Performance measurement with `time`
- File downloading with `wget`
- Text processing optimization
- Parallel processing with `xargs -P`
- Statistical analysis (median calculation)

**Prerequisites**: Internet connection for file download
**How to Run**:
```bash
chmod +x 05-speed_demo.sh
./05-speed_demo.sh [filename]  # Optional: specify custom file
```

#### 7. `06-parallel_gzip.sh` - Parallel Processing
**Purpose**: Demonstrate sequential vs parallel compression
**Key Concepts**:
- Parallel execution with `xargs -P`
- Performance timing and comparison
- CPU core utilization with `nproc`
- File compression workflows

**Prerequisites**: Internet connection for corpus download
**How to Run**:
```bash
chmod +x 06-parallel_gzip.sh
./06-parallel_gzip.sh [files...]  # Optional: specify files to compress
```

### **Phase 3: Text Processing and NLP Applications**

#### 8. `07-grep_examples.sh` - Advanced Text Search
**Purpose**: Comprehensive guide to grep command usage
**Key Concepts**:
- Basic and advanced grep patterns
- Case-insensitive search (-i)
- Recursive search (-r)
- Line numbering (-n)
- Count matches (-c)
- Inverted matching (-v)
- Regular expressions
- Whole-word matching (-w)

**How to Run**:
```bash
chmod +x 07-grep_examples.sh
./07-grep_examples.sh [filename]  # Optional: specify custom file
```

#### 9. `08-nlp_pipeline.sh` - Complete NLP Workflow
**Purpose**: Orchestrate a full NLP pipeline with multiple Python scripts
**Key Concepts**:
- Command-line argument parsing
- Directory structure management
- Multi-stage pipeline execution
- Integration with Python scripts
- Timestamped output organization

**Prerequisites**: Python scripts (preprocess.py, train.py, evaluate.py)
**How to Run**:
```bash
chmod +x 08-nlp_pipeline.sh
./08-nlp_pipeline.sh --model bert-base-uncased --epochs 5 --corpus small_corpus.txt
```

#### 10. `09-sentiment_vote.sh` - Sentiment Analysis System
**Purpose**: Implement a multi-criteria sentiment analysis system
**Key Concepts**:
- Bash arrays and functions
- Text analysis with multiple criteria
- Voting mechanisms
- Emoji and keyword processing
- Statistical summarization

**Prerequisites**: demo_corpus.txt file
**How to Run**:
```bash
chmod +x 09-sentiment_vote.sh
./09-sentiment_vote.sh
```

#### 11. `10-text_preprocessing_workflow.sh` - Text Preprocessing Pipeline
**Purpose**: Demonstrate common NLP preprocessing steps
**Key Concepts**:
- Token counting and frequency analysis
- Stop-word removal
- Case normalization
- Punctuation handling
- Stemming/lemmatization
- N-gram generation
- Pipeline chaining with pipes

**Prerequisites**: 
- stopwords.txt file
- snowball stemmer (`pip install snowball`)

**How to Run**:
```bash
chmod +x 10-text_preprocessing_workflow.sh
./10-text_preprocessing_workflow.sh
```

## 🔧 Prerequisites and Setup

### System Requirements
- Unix-like operating system (Linux, macOS, or WSL on Windows)
- Bash shell (version 4.0 or higher recommended)
- Python 3.6+ with pip
- Internet connection (for some scripts that download data)

### Required Tools
Most tools are standard on Unix systems, but you may need to install:
```bash
# For stemming functionality
pip install snowball

# Standard tools (usually pre-installed)
# wget, curl, grep, awk, sed, sort, uniq, wc, xargs
```

### File Permissions
Make scripts executable before running:
```bash
chmod +x *.sh
```

## 📊 Supporting Files

### Data Files
- `small_corpus.txt` - Small text corpus for basic examples
- `demo_corpus.txt` - Sample data for sentiment analysis
- `stopwords.txt` - Common English stop words

### Python Support Files
- `preprocess.py` - Text preprocessing utilities
- `train.py` - Model training script
- `evaluate.py` - Model evaluation script
- `stopwords_creation.py` - Stop words list generator

### Configuration
- `box_dataset_link` - Links to larger datasets

## 🎓 Learning Tips

### For Beginners
1. Start with scripts 01-04 to understand basic shell syntax
2. Practice modifying variables and conditions
3. Run each script multiple times with different inputs

### For Intermediate Users
1. Focus on scripts 05-07 for performance and text processing
2. Experiment with different file sizes and patterns
3. Try modifying the benchmarking parameters

### For Advanced Users
1. Study scripts 08-10 for real-world NLP applications
2. Integrate these patterns into your own projects
3. Extend the pipelines with additional processing steps

## 🚀 Practice Exercises

The `Class Ex/` directory contains:
- `Class_Ex_Linux.txt` - Practice problems and exercises
- `Class_Ex_Linux_sol.txt` - Solutions and explanations

Work through these exercises after completing the main scripts to reinforce your learning.

## 🔍 Troubleshooting

### Common Issues
1. **Permission Denied**: Run `chmod +x script_name.sh`
2. **Command Not Found**: Ensure required tools are installed
3. **File Not Found**: Check that data files are in the correct directory
4. **Network Issues**: Some scripts require internet access for downloads

### Getting Help
- Use `man command_name` for detailed command documentation
- Run scripts with `bash -x script_name.sh` for debugging
- Check script comments for specific requirements

## 📈 Next Steps

After mastering these scripts, consider:
1. Creating your own NLP pipelines
2. Integrating with cloud computing platforms
3. Scaling to larger datasets
4. Combining with containerization (Docker)
5. Implementing in production environments

---

**Note**: This learning material is designed for educational purposes in NLP courses. The scripts demonstrate both basic concepts and practical applications commonly used in natural language processing workflows.