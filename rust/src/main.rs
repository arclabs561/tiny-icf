/// Rust implementation of Nano-CNN ICF estimator
/// 
/// Ultra-fast inference with zero dependencies beyond standard library.
/// Weights are embedded in the binary at compile time.

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Serialize, Deserialize)]
struct ModelWeights {
    emb: Vec<Vec<f32>>,      // [256, 16]
    conv_w: Vec<Vec<Vec<f32>>>, // [32, 16, 5]
    conv_b: Vec<f32>,        // [32]
    head_w: Vec<Vec<f32>>,   // [1, 32]
    head_b: Vec<f32>,        // [1]
    metadata: Metadata,
}

#[derive(Debug, Serialize, Deserialize)]
struct Metadata {
    vocab_size: usize,
    emb_dim: usize,
    conv_channels: usize,
    kernel_size: usize,
    stride: usize,
    max_length: usize,
}

struct NanoModel {
    emb: Vec<Vec<f32>>,
    conv_w: Vec<Vec<Vec<f32>>>,
    conv_b: Vec<f32>,
    head_w: Vec<f32>,
    head_b: f32,
    stride: usize,
    kernel_size: usize,
}

impl NanoModel {
    fn from_weights(weights: ModelWeights) -> Self {
        Self {
            emb: weights.emb,
            conv_w: weights.conv_w,
            conv_b: weights.conv_b,
            head_w: weights.head_w[0].clone(), // Flatten [1, 32] -> [32]
            head_b: weights.head_b[0],
            stride: weights.metadata.stride,
            kernel_size: weights.metadata.kernel_size,
        }
    }
    
    fn forward(&self, text: &str) -> f32 {
        let bytes = text.as_bytes();
        let len = bytes.len().min(20); // Max length
        
        // 1. Embedding lookup
        let mut embedded = Vec::new();
        for &byte in bytes.iter().take(len) {
            embedded.push(self.emb[byte as usize].clone());
        }
        
        // Pad to max_length if needed
        while embedded.len() < 20 {
            embedded.push(vec![0.0; 16]); // Zero padding
        }
        
        // 2. Strided Convolution
        let out_len = (len + self.kernel_size - 1) / self.stride;
        let mut conv_out = vec![vec![0.0; out_len]; 32];
        
        for out_idx in 0..out_len {
            let input_start = out_idx * self.stride;
            
            for filter_idx in 0..32 {
                let mut sum = self.conv_b[filter_idx];
                
                // Convolve over kernel
                for k in 0..self.kernel_size {
                    let input_pos = input_start as i32 + k as i32 - (self.kernel_size / 2) as i32;
                    
                    if input_pos >= 0 && (input_pos as usize) < embedded.len() {
                        let emb_vec = &embedded[input_pos as usize];
                        
                        // Dot product: embedding * kernel weights
                        for dim in 0..16 {
                            sum += emb_vec[dim] * self.conv_w[filter_idx][dim][k];
                        }
                    }
                }
                
                // ReLU
                conv_out[filter_idx][out_idx] = sum.max(0.0);
            }
        }
        
        // 3. Global Max Pooling
        let mut pooled = vec![0.0; 32];
        for filter_idx in 0..32 {
            pooled[filter_idx] = conv_out[filter_idx]
                .iter()
                .copied()
                .fold(0.0, f32::max);
        }
        
        // 4. Linear head
        let mut logit = self.head_b;
        for i in 0..32 {
            logit += pooled[i] * self.head_w[i];
        }
        
        // 5. Sigmoid
        1.0 / (1.0 + (-logit).exp())
    }
}

#[derive(Parser)]
#[command(name = "icf-estimator")]
#[command(about = "Ultra-fast ICF score estimator")]
struct Args {
    /// Path to weights JSON file
    #[arg(short, long, default_value = "nano_weights.json")]
    weights: String,
    
    /// Words to score (space-separated)
    words: Vec<String>,
}

fn main() {
    let args = Args::parse();
    
    // Load weights
    let weights_json = fs::read_to_string(&args.weights)
        .expect("Failed to read weights file");
    let weights: ModelWeights = serde_json::from_str(&weights_json)
        .expect("Failed to parse weights JSON");
    
    let model = NanoModel::from_weights(weights);
    
    // Score words
    if args.words.is_empty() {
        eprintln!("No words provided. Usage: icf-estimator <word1> <word2> ...");
        std::process::exit(1);
    }
    
    println!("{:<20} {:<10} {}", "Word", "ICF Score", "Interpretation");
    println!("{}", "-".repeat(60));
    
    for word in args.words {
        let score = model.forward(&word);
        let interpretation = if score < 0.2 {
            "Very Common"
        } else if score < 0.5 {
            "Common"
        } else if score < 0.8 {
            "Rare"
        } else {
            "Very Rare"
        };
        
        println!("{:<20} {:<10.4} {}", word, score, interpretation);
    }
}

