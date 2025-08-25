#!/bin/bash

# XAI Loan Eligibility System Startup Script
# This script sets up and starts the complete system

set -e  # Exit on any error

echo "🚀 XAI Loan Eligibility System Startup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_step "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi

    print_status "Docker is ready ✓"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_step "Checking Docker Compose..."
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please install Docker Compose and try again."
        exit 1
    fi
    print_status "Docker Compose is ready ✓"
}

# Create necessary directories
setup_directories() {
    print_step "Setting up directories..."
    
    # Create directories if they don't exist
    mkdir -p static
    mkdir -p data
    mkdir -p backups
    mkdir -p ollama_data
    
    # Set permissions
    chmod 755 static data backups ollama_data
    
    print_status "Directories created and configured ✓"
}

# Check required files
check_files() {
    print_step "Checking required files..."
    
    required_files=("main.py" "requirements.txt" "Dockerfile" "docker-compose.yml" "static/index.html")
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_error "Missing required files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        print_error "Please ensure all required files are present and try again."
        exit 1
    fi
    
    print_status "All required files present ✓"
}

# Stop existing containers
stop_existing() {
    print_step "Stopping existing containers..."
    
    if docker-compose ps -q 2>/dev/null | grep -q .; then
        print_status "Stopping existing containers..."
        docker-compose down
    else
        print_status "No existing containers to stop ✓"
    fi
}

# Build and start services
start_services() {
    print_step "Building and starting services..."
    print_status "This may take a few minutes on first run (downloading models)..."
    
    # Build and start in detached mode
    docker-compose up --build -d
    
    print_status "Services started ✓"
}

# Wait for services to be ready
wait_for_services() {
    print_step "Waiting for services to be ready..."
    
    # Wait for main app
    print_status "Waiting for main application..."
    max_attempts=30
    attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_status "Main application is ready ✓"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            print_warning "Main application health check timeout after ${max_attempts} attempts"
            print_warning "The service might still be starting up..."
            break
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    # Wait for Phi-3 model
    print_status "Waiting for Phi-3 model to load..."
    attempt=1
    
    while [[ $attempt -le 60 ]]; do  # Longer timeout for model loading
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_status "Phi-3 service is ready ✓"
            break
        fi
        
        if [[ $attempt -eq 60 ]]; then
            print_warning "Phi-3 service timeout after 60 attempts"
            print_warning "The model might still be downloading..."
            break
        fi
        
        if [[ $((attempt % 10)) -eq 0 ]]; then
            echo -n " (${attempt}s)"
        else
            echo -n "."
        fi
        sleep 1
        ((attempt++))
    done
}

# Display service status
show_status() {
    print_step "Service Status"
    echo
    docker-compose ps
    echo
}

# Display access information
show_access_info() {
    print_step "Access Information"
    echo
    print_status "🌐 Web Application: http://localhost:8000"
    print_status "📚 API Documentation: http://localhost:8000/docs"
    print_status "❤️ Health Check: http://localhost:8000/health"
    print_status "🧠 Phi-3 API: http://localhost:11434"
    echo
    print_status "Use 'docker-compose logs -f' to view logs"
    print_status "Use 'docker-compose down' to stop all services"
    print_status "Use './test_api.py' to run API tests"
    echo
}

# Run tests (optional)
run_tests() {
    if [[ "$1" == "--test" ]]; then
        print_step "Running API tests..."
        if [[ -f "test_api.py" ]]; then
            python3 test_api.py
        else
            print_warning "test_api.py not found. Skipping tests."
        fi
    fi
}

# Main execution
main() {
    echo
    check_docker
    check_docker_compose
    setup_directories
    check_files
    stop_existing
    start_services
    wait_for_services
    show_status
    show_access_info
    run_tests "$1"
    
    print_step "🎉 XAI Loan Eligibility System is ready!"
    echo
    print_status "Visit http://localhost:8000 to start using the application"
}

# Handle script arguments
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "XAI Loan Eligibility System Startup Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --test    Run API tests after startup"
    echo "  --help    Show this help message"
    echo
    echo "This script will:"
    echo "1. Check system requirements"
    echo "2. Set up necessary directories"
    echo "3. Build and start Docker services"
    echo "4. Wait for services to be ready"
    echo "5. Display access information"
    exit 0
fi

# Run main function with all arguments
main "$@"