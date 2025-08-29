# Carregando bibliotecas necessárias
library(torch)
library(data.table)
library(dplyr)
library(lubridate)

# ========================================
# COMPONENTES AUXILIARES
# ========================================

# Positional Encoding
positional_encoding <- nn_module(
  "PositionalEncoding",
  initialize = function(d_model, max_len = 5000) {
    self$d_model <- d_model
    
    # Criar positional encoding
    pe <- torch_zeros(max_len, d_model)
    position <- torch_arange(0, max_len)$unsqueeze(2)$float()
    
    div_term <- torch_exp(torch_arange(0, d_model, 2)$float() * 
                         -(log(10000.0) / d_model))
    
    pe[, seq(1, d_model, 2)] <- torch_sin(position * div_term)
    if (d_model %% 2 == 0) {
      pe[, seq(2, d_model, 2)] <- torch_cos(position * div_term)
    } else {
      pe[, seq(2, d_model-1, 2)] <- torch_cos(position * div_term[1:(length(div_term)-1)])
    }
    
    self$register_buffer('pe', pe$unsqueeze(1))
  },
  forward = function(x) {
    seq_len <- x$shape[2]
    return(x + self$pe[1:seq_len, , ]$transpose(1, 2))
  }
)

# ========================================
# COMPONENTES MELHORADOS DO TFT
# ========================================

# 1. Gated Linear Unit (GLU) - Melhorado
glu_layer <- nn_module(
  "GLU",
  initialize = function(input_size, hidden_size = NULL, dropout = 0.1) {
    if (is.null(hidden_size)) hidden_size <- input_size
    self$linear <- nn_linear(input_size, hidden_size * 2)
    self$dropout <- nn_dropout(dropout)
    self$sigmoid <- nn_sigmoid()
  },
  forward = function(x) {
    x <- self$linear(x)
    x <- self$dropout(x)
    gate <- self$sigmoid(x[, , 1:(dim(x)[3] %/% 2)])
    value <- x[, , ((dim(x)[3] %/% 2) + 1):dim(x)[3]]
    return(gate * value)
  }
)

# 2. Gated Residual Network (GRN) - Melhorado
grn_layer <- nn_module(
  "GRN",
  initialize = function(input_size, hidden_size = NULL, output_size = NULL,
                       dropout = 0.1, context_size = NULL, use_batch_norm = TRUE) {
    if (is.null(hidden_size)) hidden_size <- input_size
    if (is.null(output_size)) output_size <- input_size
    
    self$input_size <- input_size
    self$hidden_size <- hidden_size
    self$output_size <- output_size
    self$context_size <- context_size
    
    self$fc1 <- nn_linear(input_size, hidden_size)
    self$elu <- nn_elu()
    self$fc2 <- nn_linear(hidden_size, hidden_size)
    self$dropout <- nn_dropout(dropout)
    self$gate <- nn_linear(hidden_size, output_size)
    
    if (use_batch_norm) {
      self$batch_norm <- nn_batch_norm1d(output_size)
    } else {
      self$batch_norm <- NULL
    }
    self$layer_norm <- nn_layer_norm(output_size)
    
    if (!is.null(context_size)) {
      self$context_projection <- nn_linear(context_size, hidden_size, bias = FALSE)
    }
    
    # Skip connection projection se necessário
    if (input_size != output_size) {
      self$skip_projection <- nn_linear(input_size, output_size)
    } else {
      self$skip_projection <- NULL
    }
  },
  forward = function(x, context = NULL) {
    # Projeção principal
    a <- self$fc1(x)
    if (!is.null(context) && !is.null(self$context_projection)) {
      a <- a + self$context_projection(context)
    }
    a <- self$elu(a)
    a <- self$fc2(a)
    a <- self$dropout(a)
    
    # Gating
    g <- torch_sigmoid(self$gate(a))
    
    # Skip connection com projeção se necessário
    if (!is.null(self$skip_projection)) {
      skip <- self$skip_projection(x)
    } else {
      skip <- x
    }
    
    # Aplicar gate e residual connection
    output <- skip + g * a
    
    # Batch normalization (se 3D, reshape temporariamente)
    if (!is.null(self$batch_norm) && length(output$shape) == 3) {
      batch_size <- output$shape[1]
      seq_len <- output$shape[2]
      feature_size <- output$shape[3]
      
      output_reshaped <- output$view(c(batch_size * seq_len, feature_size))
      output_reshaped <- self$batch_norm(output_reshaped)
      output <- output_reshaped$view(c(batch_size, seq_len, feature_size))
    } else if (!is.null(self$batch_norm)) {
      output <- self$batch_norm(output)
    }
    
    # Layer normalization
    output <- self$layer_norm(output)
    
    return(output)
  }
)

# 3. Variable Selection Network (VSN) - Melhorado
variable_selection <- nn_module(
  "VariableSelection",
  initialize = function(input_size, num_vars, hidden_size, dropout = 0.1) {
    self$input_size <- input_size
    self$num_vars <- num_vars
    self$hidden_size <- hidden_size
    
    # GRNs para cada variável
    self$single_var_grns <- nn_module_list()
    for (i in 1:num_vars) {
      self$single_var_grns$append(
        grn_layer(input_size, hidden_size, hidden_size, dropout)
      )
    }
    
    # GRN para seleção de variáveis
    self$var_selection_grn <- grn_layer(
      input_size * num_vars, hidden_size, num_vars, dropout
    )
    
    self$softmax <- nn_softmax(dim = -1)
  },
  forward = function(x) {
    # x: [batch, time, input_size * num_vars]
    batch_size <- x$shape[1]
    time_steps <- x$shape[2]
    
    # Separar e processar cada variável
    var_outputs <- list()
    for (i in 1:self$num_vars) {
      start_idx <- (i - 1) * self$input_size + 1
      end_idx <- i * self$input_size
      var_input <- x[, , start_idx:end_idx]
      var_outputs[[i]] <- self$single_var_grns[[i]](var_input)
    }
    
    # Concatenar para seleção
    concatenated <- torch_cat(var_outputs, dim = -1)
    
    # Pesos de seleção
    selection_weights <- self$var_selection_grn(x)
    selection_weights <- self$softmax(selection_weights)
    
    # Aplicar pesos e combinar
    selected_vars <- torch_zeros_like(var_outputs[[1]])
    for (i in 1:self$num_vars) {
      weight <- selection_weights[, , i]$unsqueeze(-1)
      selected_vars <- selected_vars + weight * var_outputs[[i]]
    }
    
    return(list(
      selected = selected_vars, 
      weights = selection_weights,
      individual_vars = var_outputs
    ))
  }
)

# 4. Multi-Head Attention - Melhorado
multi_head_attention <- nn_module(
  "MultiHeadAttention",
  initialize = function(d_model, num_heads, dropout = 0.1, use_relative_pos = TRUE) {
    self$d_model <- d_model
    self$num_heads <- num_heads
    self$head_dim <- d_model %/% num_heads
    self$use_relative_pos <- use_relative_pos
    
    self$q_linear <- nn_linear(d_model, d_model)
    self$k_linear <- nn_linear(d_model, d_model)
    self$v_linear <- nn_linear(d_model, d_model)
    self$out_linear <- nn_linear(d_model, d_model)
    self$dropout <- nn_dropout(dropout)
    
    self$scale <- sqrt(self$head_dim)
    
    # Relative positional encoding
    if (use_relative_pos) {
      self$relative_pos_embedding <- nn_embedding(512, self$head_dim)
    }
  },
  forward = function(query, key, value, mask = NULL) {
    batch_size <- query$shape[1]
    seq_len <- query$shape[2]
    
    # Projeções lineares
    Q <- self$q_linear(query)
    K <- self$k_linear(key)
    V <- self$v_linear(value)
    
    # Reshape para multi-head
    Q <- Q$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2, 3)
    K <- K$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2, 3)
    V <- V$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2, 3)
    
    # Attention scores
    scores <- torch_matmul(Q, K$transpose(-2, -1)) / self$scale
    
    # Adicionar relative positional encoding
    if (self$use_relative_pos) {
      relative_positions <- torch_arange(seq_len)$unsqueeze(1) - torch_arange(seq_len)$unsqueeze(2)
      relative_positions <- torch_clamp(relative_positions + 256, 0, 511)
      relative_pos_emb <- self$relative_pos_embedding(relative_positions)
      relative_scores <- torch_einsum("bhid,jkd->bhijk", list(Q, relative_pos_emb))
      scores <- scores + relative_scores$sum(-1)
    }
    
    if (!is.null(mask)) {
      scores <- scores$masked_fill(mask == 0, -1e9)
    }
    
    attention_weights <- torch_softmax(scores, dim = -1)
    attention_weights <- self$dropout(attention_weights)
    
    # Apply attention
    context <- torch_matmul(attention_weights, V)
    context <- context$transpose(2, 3)$contiguous()$view(
      c(batch_size, seq_len, self$d_model)
    )
    
    output <- self$out_linear(context)
    
    return(list(output = output, attention = attention_weights))
  }
)

# 5. Locality Enhancement Layer
locality_enhancement <- nn_module(
  "LocalityEnhancement",
  initialize = function(hidden_size, kernel_sizes = c(3, 5, 7), dropout = 0.1) {
    self$hidden_size <- hidden_size
    self$kernel_sizes <- kernel_sizes
    
    # Convoluções 1D para diferentes escalas temporais
    self$conv_layers <- nn_module_list()
    for (k in kernel_sizes) {
      conv_block <- nn_sequential(
        nn_conv1d(hidden_size, hidden_size, kernel_size = k, padding = k %/% 2),
        nn_batch_norm1d(hidden_size),
        nn_relu(),
        nn_dropout(dropout)
      )
      self$conv_layers$append(conv_block)
    }
    
    # Projeção final
    self$output_projection <- nn_linear(hidden_size * length(kernel_sizes), hidden_size)
    self$layer_norm <- nn_layer_norm(hidden_size)
  },
  forward = function(x) {
    # x: [batch, seq_len, hidden_size]
    # Conv1d espera [batch, channels, seq_len]
    x_conv <- x$transpose(2, 3)
    
    conv_outputs <- list()
    for (i in seq_along(self$kernel_sizes)) {
      conv_out <- self$conv_layers[[i]](x_conv)
      conv_outputs[[i]] <- conv_out$transpose(2, 3)  # Back to [batch, seq_len, hidden_size]
    }
    
    # Concatenar outputs das diferentes escalas
    concatenated <- torch_cat(conv_outputs, dim = -1)
    
    # Projeção final
    output <- self$output_projection(concatenated)
    
    # Residual connection
    output <- self$layer_norm(output + x)
    
    return(output)
  }
)

# 6. Quantile Output Layer
quantile_output_layer <- nn_module(
  "QuantileOutput",
  initialize = function(hidden_size, output_size, prediction_horizon, 
                       quantiles = c(0.1, 0.25, 0.5, 0.75, 0.9)) {
    self$quantiles <- quantiles
    self$num_quantiles <- length(quantiles)
    self$output_size <- output_size
    self$prediction_horizon <- prediction_horizon
    
    # Camadas separadas para cada quantil
    self$quantile_layers <- nn_module_list()
    for (i in seq_along(quantiles)) {
      layer <- nn_sequential(
        grn_layer(hidden_size, hidden_size, hidden_size),
        nn_linear(hidden_size, output_size * prediction_horizon)
      )
      self$quantile_layers$append(layer)
    }
  },
  forward = function(x) {
    batch_size <- x$shape[1]
    
    quantile_outputs <- list()
    for (i in seq_along(self$quantiles)) {
      output <- self$quantile_layers[[i]](x)
      output <- output$view(c(batch_size, self$prediction_horizon, self$output_size))
      quantile_outputs[[i]] <- output
    }
    
    # Stack quantiles: [batch, prediction_horizon, output_size, num_quantiles]
    stacked_outputs <- torch_stack(quantile_outputs, dim = -1)
    
    return(stacked_outputs)
  }
)

# ========================================
# EMBEDDING LAYERS PARA VARIÁVEIS CATEGÓRICAS
# ========================================

categorical_embedding <- nn_module(
  "CategoricalEmbedding",
  initialize = function(categorical_cardinalities, embedding_dim = NULL) {
    self$categorical_cardinalities <- categorical_cardinalities
    self$num_categorical <- length(categorical_cardinalities)
    
    if (is.null(embedding_dim)) {
      embedding_dim <- pmin(50, (categorical_cardinalities + 1) %/% 2)
    }
    
    self$embedding_layers <- nn_module_list()
    self$embedding_dims <- embedding_dim
    
    for (i in seq_along(categorical_cardinalities)) {
      embedding_layer <- nn_embedding(
        categorical_cardinalities[i],
        embedding_dim[i]
      )
      self$embedding_layers$append(embedding_layer)
    }
    
    self$total_embedding_dim <- sum(embedding_dim)
  },
  forward = function(categorical_inputs) {
    # categorical_inputs: list of tensors, one for each categorical variable
    embeddings <- list()
    
    for (i in seq_along(categorical_inputs)) {
      emb <- self$embedding_layers[[i]](categorical_inputs[[i]])
      embeddings[[i]] <- emb
    }
    
    # Concatenar todos os embeddings
    if (length(embeddings) > 1) {
      concatenated <- torch_cat(embeddings, dim = -1)
    } else {
      concatenated <- embeddings[[1]]
    }
    
    return(concatenated)
  }
)

# ========================================
# MODELO TFT COMPLETO E MELHORADO
# ========================================

temporal_fusion_transformer_enhanced <- nn_module(
  "TemporalFusionTransformerEnhanced",
  initialize = function(
    # Configurações básicas
    input_size,
    output_size = 1,
    hidden_size = 128,
    num_heads = 8,
    num_layers = 3,
    dropout = 0.1,
    prediction_horizon = 1,
    
    # Configurações de variáveis
    num_static_vars = 0,
    num_time_varying_vars = 1,
    categorical_cardinalities = NULL,
    
    # Configurações avançadas
    use_quantile_forecasting = TRUE,
    quantiles = c(0.1, 0.25, 0.5, 0.75, 0.9),
    use_locality_enhancement = TRUE,
    use_relative_attention = TRUE,
    
    # Configurações específicas
    encoder_length = 24,
    max_sequence_length = 256
  ) {
    
    self$input_size <- input_size
    self$output_size <- output_size
    self$hidden_size <- hidden_size
    self$num_heads <- num_heads
    self$prediction_horizon <- prediction_horizon
    self$encoder_length <- encoder_length
    self$use_quantile_forecasting <- use_quantile_forecasting
    self$quantiles <- quantiles
    
    # ========================================
    # EMBEDDING LAYERS
    # ========================================
    
    # Embeddings categóricos
    if (!is.null(categorical_cardinalities)) {
      self$categorical_embedding <- categorical_embedding(categorical_cardinalities)
      categorical_input_size <- self$categorical_embedding$total_embedding_dim
    } else {
      self$categorical_embedding <- NULL
      categorical_input_size <- 0
    }
    
    # Static variable embeddings
    self$static_embedding <- if (num_static_vars > 0) {
      nn_linear(num_static_vars + categorical_input_size, hidden_size)
    } else if (categorical_input_size > 0) {
      nn_linear(categorical_input_size, hidden_size)
    } else NULL
    
    # Time-varying embeddings
    self$time_varying_embedding <- nn_linear(num_time_varying_vars, hidden_size)
    
    # Positional encoding
    self$positional_encoding <- positional_encoding(hidden_size, max_sequence_length)
    
    # ========================================
    # VARIABLE SELECTION
    # ========================================
    
    # Static variable selection
    self$static_selection <- if (!is.null(self$static_embedding)) {
      variable_selection(hidden_size, 1, hidden_size, dropout)
    } else NULL
    
    # Temporal variable selection (past)
    self$temporal_selection_past <- variable_selection(
      hidden_size, 1, hidden_size, dropout
    )
    
    # Temporal variable selection (future known)
    self$temporal_selection_future <- variable_selection(
      hidden_size, 1, hidden_size, dropout
    )
    
    # ========================================
    # ENCODER-DECODER ARCHITECTURE
    # ========================================
    
    # Encoder LSTM para dados históricos
    self$encoder_lstm <- nn_lstm(
      input_size = hidden_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      batch_first = TRUE,
      dropout = dropout,
      bidirectional = FALSE
    )
    
    # Decoder LSTM para previsões futuras
    self$decoder_lstm <- nn_lstm(
      input_size = hidden_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      batch_first = TRUE,
      dropout = dropout
    )
    
    # ========================================
    # ATTENTION E LOCALITY
    # ========================================
    
    # Multi-head attention
    self$self_attention <- multi_head_attention(
      hidden_size, num_heads, dropout, use_relative_attention
    )
    
    # Locality enhancement
    if (use_locality_enhancement) {
      self$locality_enhancement <- locality_enhancement(hidden_size, dropout = dropout)
    } else {
      self$locality_enhancement <- NULL
    }
    
    # ========================================
    # GATED RESIDUAL NETWORKS
    # ========================================
    
    # Post-attention processing
    self$post_attention_grn <- grn_layer(hidden_size, hidden_size, hidden_size, dropout)
    
    # Static enrichment
    self$static_enrichment <- if (!is.null(self$static_embedding)) {
      grn_layer(hidden_size, hidden_size, hidden_size, dropout, hidden_size)
    } else NULL
    
    # Temporal fusion
    self$temporal_fusion_grn <- grn_layer(
      hidden_size * 2, hidden_size, hidden_size, dropout
    )
    
    # ========================================
    # OUTPUT LAYERS
    # ========================================
    
    if (use_quantile_forecasting) {
      self$output_layer <- quantile_output_layer(
        hidden_size, output_size, prediction_horizon, quantiles
      )
    } else {
      self$output_layer <- nn_sequential(
        grn_layer(hidden_size, hidden_size, hidden_size, dropout),
        nn_linear(hidden_size, output_size * prediction_horizon)
      )
    }
    
  },
  
  forward = function(x_static = NULL, x_past, x_future_known = NULL, 
                    categorical_static = NULL, categorical_past = NULL,
                    categorical_future = NULL) {
    
    batch_size <- x_past$shape[1]
    past_seq_len <- x_past$shape[2]
    future_seq_len <- if (!is.null(x_future_known)) x_future_known$shape[2] else self$prediction_horizon
    
    # ========================================
    # EMBEDDING PROCESSING
    # ========================================
    
    # Process categorical variables
    categorical_emb_static <- NULL
    categorical_emb_past <- NULL
    categorical_emb_future <- NULL
    
    if (!is.null(self$categorical_embedding)) {
      if (!is.null(categorical_static)) {
        categorical_emb_static <- self$categorical_embedding(categorical_static)
      }
      if (!is.null(categorical_past)) {
        categorical_emb_past <- self$categorical_embedding(categorical_past)
      }
      if (!is.null(categorical_future)) {
        categorical_emb_future <- self$categorical_embedding(categorical_future)
      }
    }
    
    # Static features
    static_features <- NULL
    if (!is.null(x_static) || !is.null(categorical_emb_static)) {
      static_input <- NULL
      if (!is.null(x_static) && !is.null(categorical_emb_static)) {
        static_input <- torch_cat(list(x_static, categorical_emb_static), dim = -1)
      } else if (!is.null(x_static)) {
        static_input <- x_static
      } else {
        static_input <- categorical_emb_static
      }
      
      static_emb <- self$static_embedding(static_input)
      static_selected <- self$static_selection(static_emb$unsqueeze(2))
      static_features <- static_selected$selected$squeeze(2)
    }
    
    # Past temporal features
    past_temporal_input <- x_past
    if (!is.null(categorical_emb_past)) {
      past_temporal_input <- torch_cat(list(x_past, categorical_emb_past), dim = -1)
      # Adjust embedding layer if needed
      if (past_temporal_input$shape[3] != self$time_varying_embedding$in_features) {
        # Create a new embedding layer for this size if needed
        # For simplicity, we'll just use the continuous part
        past_temporal_input <- x_past
      }
    }
    
    past_temporal_emb <- self$time_varying_embedding(past_temporal_input)
    past_temporal_emb <- self$positional_encoding(past_temporal_emb)
    
    past_selected <- self$temporal_selection_past(past_temporal_emb)
    past_features <- past_selected$selected
    
    # Future temporal features (if available)
    future_features <- NULL
    if (!is.null(x_future_known)) {
      future_temporal_input <- x_future_known
      if (!is.null(categorical_emb_future)) {
        future_temporal_input <- torch_cat(list(x_future_known, categorical_emb_future), dim = -1)
        if (future_temporal_input$shape[3] != self$time_varying_embedding$in_features) {
          future_temporal_input <- x_future_known
        }
      }
      
      future_temporal_emb <- self$time_varying_embedding(future_temporal_input)
      # Adjust positional encoding for future
      future_pos_emb <- self$positional_encoding$pe[(past_seq_len + 1):(past_seq_len + future_seq_len), , ]
      future_temporal_emb <- future_temporal_emb + future_pos_emb$transpose(1, 2)
      
      future_selected <- self$temporal_selection_future(future_temporal_emb)
      future_features <- future_selected$selected
    }
    
    # ========================================
    # ENCODER-DECODER PROCESSING
    # ========================================
    
    # Encoder: Process historical data
    encoder_output <- self$encoder_lstm(past_features)
    encoder_hidden <- encoder_output[[1]]  # [batch, past_seq_len, hidden_size]
    encoder_states <- encoder_output[[2]]   # (h_n, c_n)
    
    # Decoder: Process future (or generate future states)
    if (!is.null(future_features)) {
      # Use known future features
      decoder_output <- self$decoder_lstm(future_features, encoder_states)
      decoder_hidden <- decoder_output[[1]]
    } else {
      # Generate future states auto-regressively
      decoder_hidden <- list()
      decoder_state <- encoder_states
      
      # Use last encoder output as initial input
      decoder_input <- encoder_hidden[, -1, ]$unsqueeze(2)  # [batch, 1, hidden_size]
      
      for (t in 1:self$prediction_horizon) {
        decoder_step_output <- self$decoder_lstm(decoder_input, decoder_state)
        decoder_step_hidden <- decoder_step_output[[1]]
        decoder_state <- decoder_step_output[[2]]
        
        decoder_hidden[[t]] <- decoder_step_hidden
        decoder_input <- decoder_step_hidden  # Use output as next input
      }
      
      decoder_hidden <- torch_cat(decoder_hidden, dim = 2)  # [batch, future_seq_len, hidden_size]
    }
    
    # Combine encoder and decoder outputs
    combined_hidden <- torch_cat(list(encoder_hidden, decoder_hidden), dim = 2)
    
    # ========================================
    # ATTENTION AND LOCALITY
    # ========================================
    
    # Self-attention
    attention_output <- self$self_attention(combined_hidden, combined_hidden, combined_hidden)
    attended_features <- attention_output$output
    attention_weights <- attention_output$attention
    
    # Locality enhancement
    if (!is.null(self$locality_enhancement)) {
      attended_features <- self$locality_enhancement(attended_features)
    }
    
    # Post-attention processing
    processed_features <- self$post_attention_grn(attended_features)
    
    # ========================================
    # STATIC ENRICHMENT
    # ========================================
    
    if (!is.null(static_features) && !is.null(self$static_enrichment)) {
      seq_len <- processed_features$shape[2]
      static_expanded <- static_features$unsqueeze(2)$expand(c(-1, seq_len, -1))
      processed_features <- self$static_enrichment(processed_features, static_expanded)
    }
    
    # ========================================
    # TEMPORAL FUSION
    # ========================================
    
    # Separate past and future for fusion
    past_processed <- processed_features[, 1:past_seq_len, ]
    future_processed <- processed_features[, (past_seq_len + 1):(past_seq_len + future_seq_len), ]
    
    # Temporal fusion: combine past context with future
    past_context <- past_processed[, -1, ]$unsqueeze(2)$expand(c(-1, future_seq_len, -1))
    fused_input <- torch_cat(list(future_processed, past_context), dim = -1)
    fused_features <- self$temporal_fusion_grn(fused_input)
    
    # ========================================
    # OUTPUT GENERATION
    # ========================================
    
    # Use mean of fused features for final prediction
    final_features <- fused_features$mean(dim = 2)  # [batch, hidden_size]
    
    # Generate predictions
    if (self$use_quantile_forecasting) {
      predictions <- self$output_layer(final_features)
      # predictions: [batch, prediction_horizon, output_size, num_quantiles]
    } else {
      predictions <- self$output_layer(final_features)
      predictions <- predictions$view(c(batch_size, self$prediction_horizon, self$output_size))
    }
    
    return(list(
      predictions = predictions,
      attention_weights = attention_weights,
      temporal_weights_past = past_selected$weights,
      temporal_weights_future = if (!is.null(future_features)) future_selected$weights else NULL,
      static_weights = if (!is.null(static_features)) static_selected$weights else NULL
    ))
  }
)

# ========================================
# FUNÇÕES AUXILIARES MELHORADAS
# ========================================

# Função para detectar feriados (exemplo básico)
detect_holidays <- function(dates) {
  # Exemplo básico - você pode expandir com feriados específicos do seu país
  holidays <- c(
    as.Date("2023-01-01"), as.Date("2023-12-25"),  # Exemplo
    as.Date("2024-01-01"), as.Date("2024-12-25")
  )
  
  return(as.numeric(dates %in% holidays))
}

# Função para detectar promoções (placeholder)
detect_promotions <- function(data) {
  # Placeholder - implementar lógica específica do seu negócio
  return(rep(0, nrow(data)))
}

# Função melhorada para preparar dados
preparar_dados_vendas_completo <- function(
  data, 
  target_col, 
  seq_length = 24,
  prediction_horizon = 6,
  static_cols = NULL,
  categorical_cols = NULL,
  date_col = "data",
  include_calendar_features = TRUE,
  include_lag_features = TRUE,
  lag_periods = c(1, 7, 30),
  validation_split = 0.2,
  test_split = 0.1
) {
  
  cat("Preparando dados para TFT...\n")
  
  # Converter para data.table para eficiência
  if (!is.data.table(data)) {
    data <- as.data.table(data)
  }
  
  # ========================================
  # FEATURE ENGINEERING
  # ========================================
  
  if (include_calendar_features && date_col %in% names(data)) {
    cat("Adicionando features de calendário...\n")
    data[, `:=`(
      dia_semana = as.numeric(format(get(date_col), "%u")),
      mes = as.numeric(format(get(date_col), "%m")),
      dia_mes = as.numeric(format(get(date_col), "%d")),
      trimestre = quarter(get(date_col)),
      eh_fim_semana = as.numeric(weekdays(get(date_col)) %in% c("Saturday", "Sunday")),
      feriado = detect_holidays(get(date_col))
    )]
    
    # Adicionar à lista de colunas categóricas se não estiver
    new_categorical <- c("dia_semana", "mes", "trimestre")
    if (is.null(categorical_cols)) {
      categorical_cols <- new_categorical
    } else {
      categorical_cols <- unique(c(categorical_cols, new_categorical))
    }
  }
  
  if (include_lag_features) {
    cat("Adicionando features de lag...\n")
    for (lag in lag_periods) {
      data[, paste0(target_col, "_lag_", lag) := shift(get(target_col), lag)]
    }
    
    # Remover NAs causados por lags
    data <- data[complete.cases(data)]
  }
  
  # ========================================
  # PROCESSAMENTO DE VARIÁVEIS CATEGÓRICAS
  # ========================================
  
  categorical_mappings <- list()
  categorical_cardinalities <- NULL
  
  if (!is.null(categorical_cols)) {
    cat("Processando variáveis categóricas...\n")
    categorical_cardinalities <- numeric(length(categorical_cols))
    names(categorical_cardinalities) <- categorical_cols
    
    for (i in seq_along(categorical_cols)) {
      col <- categorical_cols[i]
      if (col %in% names(data)) {
        # Converter para fator e depois para numérico (0-indexado para embeddings)
        data[, (col) := as.factor(get(col))]
        levels_map <- levels(data[[col]])
        categorical_mappings[[col]] <- levels_map
        data[, (col) := as.numeric(get(col)) - 1]  # 0-indexado
        categorical_cardinalities[col] <- length(levels_map)
      }
    }
  }
  
  # ========================================
  # NORMALIZAÇÃO
  # ========================================
  
  # Identificar colunas numéricas (excluindo categóricas e target)
  numeric_cols <- setdiff(names(data), c(categorical_cols, date_col))
  
  # Parâmetros de normalização
  normalization_params <- list()
  
  for (col in numeric_cols) {
    if (col %in% names(data) && is.numeric(data[[col]])) {
      mean_val <- mean(data[[col]], na.rm = TRUE)
      sd_val <- sd(data[[col]], na.rm = TRUE)
      
      normalization_params[[col]] <- list(mean = mean_val, sd = sd_val)
      
      # Normalizar (exceto o target que será normalizado separadamente)
      if (col != target_col) {
        data[, (col) := (get(col) - mean_val) / (sd_val + 1e-8)]
      }
    }
  }
  
  # Normalização especial para o target
  target_mean <- mean(data[[target_col]], na.rm = TRUE)
  target_sd <- sd(data[[target_col]], na.rm = TRUE)
  normalization_params[[target_col]] <- list(mean = target_mean, sd = target_sd)
  data[, (target_col) := (get(target_col) - target_mean) / (target_sd + 1e-8)]
  
  # ========================================
  # CRIAÇÃO DE SEQUÊNCIAS
  # ========================================
  
  cat("Criando sequências temporais...\n")
  
  # Identificar colunas para cada tipo
  target_idx <- which(names(data) == target_col)
  static_indices <- if (!is.null(static_cols)) which(names(data) %in% static_cols) else NULL
  categorical_indices <- if (!is.null(categorical_cols)) which(names(data) %in% categorical_cols) else NULL
  
  # Colunas time-varying (todas exceto static e categóricas)
  time_varying_cols <- setdiff(names(data), c(static_cols, categorical_cols, date_col))
  time_varying_indices <- which(names(data) %in% time_varying_cols)
  
  # Converter para matriz
  data_matrix <- as.matrix(data[, !date_col, with = FALSE])
  
  sequences_past <- list()
  sequences_future <- list()
  targets <- list()
  static_features <- list()
  categorical_features_past <- list()
  categorical_features_static <- list()
  
  for (i in 1:(nrow(data_matrix) - seq_length - prediction_horizon + 1)) {
    # Sequência passada
    past_seq <- data_matrix[i:(i + seq_length - 1), time_varying_indices, drop = FALSE]
    sequences_past[[length(sequences_past) + 1]] <- past_seq
    
    # Sequência futura (conhecida) - assumindo que temos algumas variáveis conhecidas
    # Para este exemplo, vamos criar uma versão simplificada
    future_seq <- matrix(0, nrow = prediction_horizon, ncol = length(time_varying_indices))
    sequences_future[[length(sequences_future) + 1]] <- future_seq
    
    # Target
    target_seq <- data_matrix[(i + seq_length):(i + seq_length + prediction_horizon - 1), 
                             target_idx, drop = FALSE]
    targets[[length(targets) + 1]] <- target_seq
    
    # Static features (usar do primeiro timestep da sequência)
    if (!is.null(static_indices)) {
      static_feat <- data_matrix[i, static_indices, drop = FALSE]
      static_features[[length(static_features) + 1]] <- static_feat
    }
    
    # Categorical features
    if (!is.null(categorical_indices)) {
      # Categorical past
      cat_past <- data_matrix[i:(i + seq_length - 1), categorical_indices, drop = FALSE]
      categorical_features_past[[length(categorical_features_past) + 1]] <- cat_past
      
      # Categorical static (usar do primeiro timestep)
      cat_static <- data_matrix[i, categorical_indices, drop = FALSE]
      categorical_features_static[[length(categorical_features_static) + 1]] <- cat_static
    }
  }
  
  # Converter para tensors
  X_past <- torch_stack(lapply(sequences_past, torch_tensor))$float()
  X_future <- torch_stack(lapply(sequences_future, torch_tensor))$float()
  y <- torch_stack(lapply(targets, torch_tensor))$float()
  
  X_static <- if (length(static_features) > 0) {
    torch_stack(lapply(static_features, torch_tensor))$float()
  } else NULL
  
  categorical_past_tensor <- if (length(categorical_features_past) > 0) {
    torch_stack(lapply(categorical_features_past, torch_tensor))$long()
  } else NULL
  
  categorical_static_tensor <- if (length(categorical_features_static) > 0) {
    torch_stack(lapply(categorical_features_static, torch_tensor))$long()
  } else NULL
  
  # ========================================
  # DIVISÃO DOS DADOS
  # ========================================
  
  n_total <- X_past$shape[1]
  n_test <- as.integer(test_split * n_total)
  n_valid <- as.integer(validation_split * n_total)
  n_train <- n_total - n_test - n_valid
  
  # Índices (sequencial para preservar ordem temporal)
  train_indices <- 1:n_train
  valid_indices <- (n_train + 1):(n_train + n_valid)
  test_indices <- (n_train + n_valid + 1):n_total
  
  # Criar splits
  splits <- list(
    train = list(
      X_past = X_past[train_indices],
      X_future = X_future[train_indices],
      y = y[train_indices],
      X_static = if (!is.null(X_static)) X_static[train_indices] else NULL,
      categorical_past = if (!is.null(categorical_past_tensor)) categorical_past_tensor[train_indices] else NULL,
      categorical_static = if (!is.null(categorical_static_tensor)) categorical_static_tensor[train_indices] else NULL
    ),
    valid = list(
      X_past = X_past[valid_indices],
      X_future = X_future[valid_indices], 
      y = y[valid_indices],
      X_static = if (!is.null(X_static)) X_static[valid_indices] else NULL,
      categorical_past = if (!is.null(categorical_past_tensor)) categorical_past_tensor[valid_indices] else NULL,
      categorical_static = if (!is.null(categorical_static_tensor)) categorical_static_tensor[valid_indices] else NULL
    ),
    test = list(
      X_past = X_past[test_indices],
      X_future = X_future[test_indices],
      y = y[test_indices],
      X_static = if (!is.null(X_static)) X_static[test_indices] else NULL,
      categorical_past = if (!is.null(categorical_past_tensor)) categorical_past_tensor[test_indices] else NULL,
      categorical_static = if (!is.null(categorical_static_tensor)) categorical_static_tensor[test_indices] else NULL
    )
  )
  
  cat("Dados preparados com sucesso!\n")
  cat("Total de sequências:", n_total, "\n")
  cat("Treino:", n_train, "| Validação:", n_valid, "| Teste:", n_test, "\n")
  
  return(list(
    splits = splits,
    normalization_params = normalization_params,
    categorical_mappings = categorical_mappings,
    categorical_cardinalities = categorical_cardinalities,
    feature_names = list(
      time_varying = time_varying_cols,
      static = static_cols,
      categorical = categorical_cols,
      target = target_col
    )
  ))
}

# Função de treinamento melhorada
treinar_tft_completo <- function(
  modelo, 
  dados_preparados,
  epochs = 100, 
  lr = 0.001,
  batch_size = 32,
  weight_decay = 1e-5,
  scheduler_factor = 0.5,
  scheduler_patience = 10,
  early_stopping_patience = 20,
  gradient_clip_norm = 1.0,
  quantile_loss_weights = NULL
) {
  
  # Extrair dados
  train_data <- dados_preparados$splits$train
  valid_data <- dados_preparados$splits$valid
  
  # Configurar optimizer
  optimizer <- optim_adam(modelo$parameters, lr = lr, weight_decay = weight_decay)
  
  # Scheduler
  scheduler <- lr_reduce_on_plateau(optimizer, factor = scheduler_factor, 
                                   patience = scheduler_patience, verbose = TRUE)
  
  # Loss functions
  if (modelo$use_quantile_forecasting) {
    criterion <- function(predictions, targets, quantiles = modelo$quantiles) {
      # predictions: [batch, prediction_horizon, output_size, num_quantiles]
      # targets: [batch, prediction_horizon, output_size]
      
      if (is.null(quantile_loss_weights)) {
        weights <- rep(1.0, length(quantiles))
      } else {
        weights <- quantile_loss_weights
      }
      
      total_loss <- 0
      targets_expanded <- targets$unsqueeze(-1)$expand_as(predictions)
      
      for (i in seq_along(quantiles)) {
        q <- quantiles[i]
        pred_q <- predictions[, , , i]
        target_q <- targets
        
        error <- target_q - pred_q
        loss_q <- torch_maximum(q * error, (q - 1) * error)
        total_loss <- total_loss + weights[i] * loss_q$mean()
      }
      
      return(total_loss / length(quantiles))
    }
  } else {
    criterion <- nn_mse_loss()
  }
  
  # Histórico
  history <- list(
    train_losses = numeric(epochs),
    valid_losses = numeric(epochs),
    learning_rates = numeric(epochs)
  )
  
  # Early stopping
  best_valid_loss <- Inf
  patience_counter <- 0
  best_model_state <- NULL
  
  cat("Iniciando treinamento do TFT Enhanced...\n")
  cat("Epochs:", epochs, "| Batch size:", batch_size, "| Learning rate:", lr, "\n")
  
  for (epoch in 1:epochs) {
    # ========================================
    # TREINAMENTO
    # ========================================
    modelo$train()
    epoch_train_loss <- 0
    n_train_batches <- 0
    
    # Mini-batches para treino
    n_train_samples <- train_data$X_past$shape[1]
    train_indices <- sample(n_train_samples)
    
    for (i in seq(1, n_train_samples, batch_size)) {
      end_idx <- min(i + batch_size - 1, n_train_samples)
      batch_indices <- train_indices[i:end_idx]
      
      # Preparar batch
      X_past_batch <- train_data$X_past[batch_indices]
      X_future_batch <- train_data$X_future[batch_indices]
      y_batch <- train_data$y[batch_indices]
      
      X_static_batch <- if (!is.null(train_data$X_static)) {
        train_data$X_static[batch_indices]
      } else NULL
      
      categorical_past_batch <- if (!is.null(train_data$categorical_past)) {
        train_data$categorical_past[batch_indices]
      } else NULL
      
      categorical_static_batch <- if (!is.null(train_data$categorical_static)) {
        train_data$categorical_static[batch_indices]
      } else NULL
      
      optimizer$zero_grad()
      
      # Forward pass
      output <- modelo(
        x_static = X_static_batch,
        x_past = X_past_batch,
        x_future_known = X_future_batch,
        categorical_static = categorical_static_batch,
        categorical_past = categorical_past_batch
      )
      
      predictions <- output$predictions
      
      # Calcular loss
      loss <- criterion(predictions, y_batch)
      
      # Backward pass
      loss$backward()
      
      # Gradient clipping
      nn_utils_clip_grad_norm_(modelo$parameters, gradient_clip_norm)
      
      optimizer$step()
      
      epoch_train_loss <- epoch_train_loss + loss$item()
      n_train_batches <- n_train_batches + 1
    }
    
    history$train_losses[epoch] <- epoch_train_loss / n_train_batches
    
    # ========================================
    # VALIDAÇÃO
    # ========================================
    modelo$eval()
    epoch_valid_loss <- 0
    n_valid_batches <- 0
    
    with_no_grad({
      n_valid_samples <- valid_data$X_past$shape[1]
      
      for (i in seq(1, n_valid_samples, batch_size)) {
        end_idx <- min(i + batch_size - 1, n_valid_samples)
        
        X_past_batch <- valid_data$X_past[i:end_idx]
        X_future_batch <- valid_data$X_future[i:end_idx]
        y_batch <- valid_data$y[i:end_idx]
        
        X_static_batch <- if (!is.null(valid_data$X_static)) {
          valid_data$X_static[i:end_idx]
        } else NULL
        
        categorical_past_batch <- if (!is.null(valid_data$categorical_past)) {
          valid_data$categorical_past[i:end_idx]
        } else NULL
        
        categorical_static_batch <- if (!is.null(valid_data$categorical_static)) {
          valid_data$categorical_static[i:end_idx]
        } else NULL
        
        output <- modelo(
          x_static = X_static_batch,
          x_past = X_past_batch,
          x_future_known = X_future_batch,
          categorical_static = categorical_static_batch,
          categorical_past = categorical_past_batch
        )
        
        predictions <- output$predictions
        valid_loss <- criterion(predictions, y_batch)
        
        epoch_valid_loss <- epoch_valid_loss + valid_loss$item()
        n_valid_batches <- n_valid_batches + 1
      }
    })
    
    history$valid_losses[epoch] <- epoch_valid_loss / n_valid_batches
    history$learning_rates[epoch] <- optimizer$param_groups[[1]]$lr
    
    # ========================================
    # SCHEDULER E EARLY STOPPING
    # ========================================
    
    current_valid_loss <- history$valid_losses[epoch]
    scheduler$step(current_valid_loss)
    
    # Early stopping check
    if (current_valid_loss < best_valid_loss) {
      best_valid_loss <- current_valid_loss
      patience_counter <- 0
      best_model_state <- modelo$state_dict()
    } else {
      patience_counter <- patience_counter + 1
    }
    
    # Log progresso
    if (epoch %% 5 == 0 || epoch <= 10) {
      cat(sprintf(
        "Epoch %d: Train Loss = %.6f, Valid Loss = %.6f, LR = %.2e\n", 
        epoch, history$train_losses[epoch], history$valid_losses[epoch],
        history$learning_rates[epoch]
      ))
    }
    
    # Early stopping
    if (patience_counter >= early_stopping_patience) {
      cat("Early stopping triggered at epoch", epoch, "\n")
      break
    }
  }
  
  # Restaurar melhor modelo
  if (!is.null(best_model_state)) {
    modelo$load_state_dict(best_model_state)
    cat("Modelo restaurado para melhor estado (valid loss =", best_valid_loss, ")\n")
  }
  
  # Truncar histórico se parou antes
  if (epoch < epochs) {
    history$train_losses <- history$train_losses[1:epoch]
    history$valid_losses <- history$valid_losses[1:epoch]
    history$learning_rates <- history$learning_rates[1:epoch]
  }
  
  return(list(
    history = history,
    best_valid_loss = best_valid_loss,
    final_epoch = epoch
  ))
}

# Função de previsão melhorada
prever_tft_completo <- function(modelo, dados_preparados, split = "test", 
                                return_quantiles = TRUE, return_attention = TRUE) {
  
  modelo$eval()
  
  # Selecionar dados
  if (split == "test") {
    data <- dados_preparados$splits$test
  } else if (split == "valid") {
    data <- dados_preparados$splits$valid
  } else {
    data <- dados_preparados$splits$train
  }
  
  predictions_list <- list()
  targets_list <- list()
  attention_weights_list <- list()
  
  with_no_grad({
    n_samples <- data$X_past$shape[1]
    batch_size <- 32  # Usar batch pequeno para previsão
    
    for (i in seq(1, n_samples, batch_size)) {
      end_idx <- min(i + batch_size - 1, n_samples)
      
      X_past_batch <- data$X_past[i:end_idx]
      X_future_batch <- data$X_future[i:end_idx]
      y_batch <- data$y[i:end_idx]
      
      X_static_batch <- if (!is.null(data$X_static)) {
        data$X_static[i:end_idx]
      } else NULL
      
      categorical_past_batch <- if (!is.null(data$categorical_past)) {
        data$categorical_past[i:end_idx]
      } else NULL
      
      categorical_static_batch <- if (!is.null(data$categorical_static)) {
        data$categorical_static[i:end_idx]
      } else NULL
      
      output <- modelo(
        x_static = X_static_batch,
        x_past = X_past_batch,
        x_future_known = X_future_batch,
        categorical_static = categorical_static_batch,
        categorical_past = categorical_past_batch
      )
      
      predictions_list[[length(predictions_list) + 1]] <- output$predictions$cpu()
      targets_list[[length(targets_list) + 1]] <- y_batch$cpu()
      
      if (return_attention) {
        attention_weights_list[[length(attention_weights_list) + 1]] <- output$attention_weights$cpu()
      }
    }
  })
  
  # Concatenar resultados
  all_predictions <- torch_cat(predictions_list, dim = 1)
  all_targets <- torch_cat(targets_list, dim = 1)
  
  # Converter para arrays
  predictions_array <- as.array(all_predictions)
  targets_array <- as.array(all_targets)
  
  # Desnormalizar
  target_mean <- dados_preparados$normalization_params[[dados_preparados$feature_names$target]]$mean
  target_sd <- dados_preparados$normalization_params[[dados_preparados$feature_names$target]]$sd
  
  if (modelo$use_quantile_forecasting) {
    # predictions_array: [n_samples, prediction_horizon, output_size, num_quantiles]
    for (i in 1:dim(predictions_array)[4]) {
      predictions_array[, , , i] <- predictions_array[, , , i] * target_sd + target_mean
    }
  } else {
    predictions_array <- predictions_array * target_sd + target_mean
  }
  
  targets_array <- targets_array * target_sd + target_mean
  
  result <- list(
    predictions = predictions_array,
    targets = targets_array,
    quantiles = if (modelo$use_quantile_forecasting) modelo$quantiles else NULL
  )
  
  if (return_attention && length(attention_weights_list) > 0) {
    all_attention <- torch_cat(attention_weights_list, dim = 1)
    result$attention_weights <- as.array(all_attention)
  }
  
  return(result)
}

# Função para calcular métricas
calcular_metricas_tft <- function(predictions, targets, quantiles = NULL) {
  
  if (!is.null(quantiles)) {
    # Métricas para quantile forecasting
    median_idx <- which.min(abs(quantiles - 0.5))
    point_predictions <- predictions[, , , median_idx]
  } else {
    point_predictions <- predictions
  }
  
  # Métricas básicas
  mae <- mean(abs(point_predictions - targets))
  mse <- mean((point_predictions - targets)^2)
  rmse <- sqrt(mse)
  
  # MAPE (evitando divisão por zero)
  mape <- mean(abs((targets - point_predictions) / pmax(abs(targets), 1e-8))) * 100
  
  # R²
  ss_res <- sum((targets - point_predictions)^2)
  ss_tot <- sum((targets - mean(targets))^2)
  r2 <- 1 - (ss_res / ss_tot)
  
  metrics <- list(
    MAE = mae,
    MSE = mse,
    RMSE = rmse,
    MAPE = mape,
    R2 = r2
  )
  
  # Métricas específicas de quantile
  if (!is.null(quantiles)) {
    quantile_losses <- numeric(length(quantiles))
    coverage_rates <- numeric(length(quantiles))
    
    for (i in seq_along(quantiles)) {
      q <- quantiles[i]
      pred_q <- predictions[, , , i]
      
      # Quantile loss
      error <- targets - pred_q
      quantile_losses[i] <- mean(pmax(q * error, (q - 1) * error))
      
      # Coverage rate
      coverage_rates[i] <- mean(targets <= pred_q)
    }
    
    metrics$quantile_losses <- quantile_losses
    metrics$coverage_rates <- coverage_rates
    metrics$quantiles <- quantiles
  }
  
  return(metrics)
}

# ========================================
# EXEMPLO DE USO COMPLETO
# ========================================

exemplo_uso_tft_completo <- function() {
  cat("=== EXEMPLO DE USO DO TFT COMPLETO ===\n")
  
  # ========================================
  # DADOS SIMULADOS REALISTAS
  # ========================================
  set.seed(42)
  n <- 2000
  
  # Criar série temporal mais realística para vendas
  dates <- seq(as.Date("2020-01-01"), by = "day", length.out = n)
  
  # Componentes da série
  trend <- cumsum(rnorm(n, 0.01, 0.1))
  seasonal_weekly <- 3 * sin(2 * pi * (as.numeric(dates) %% 7) / 7)
  seasonal_monthly <- 2 * sin(2 * pi * (as.numeric(format(dates, "%j"))) / 365.25)
  promotional_effect <- rbinom(n, 1, 0.1) * runif(n, 5, 15)  # Promoções esporádicas
  noise <- rnorm(n, 0, 1)
  
  vendas <- pmax(0, 100 + trend + seasonal_weekly + seasonal_monthly + promotional_effect + noise)
  
  # Variáveis adicionais
  temperatura <- 20 + 10 * sin(2 * pi * (as.numeric(format(dates, "%j"))) / 365.25) + rnorm(n, 0, 3)
  precipitacao <- pmax(0, rnorm(n, 5, 3))
  
  # Criar data frame
  data_vendas <- data.frame(
    data = dates,
    vendas = vendas,
    temperatura = temperatura,
    precipitacao = precipitacao,
    loja_id = sample(1:5, n, replace = TRUE),
    categoria_produto = sample(1:3, n, replace = TRUE),
    promocao_ativa = rbinom(n, 1, 0.1)
  )
  
  cat("Dados simulados criados:", nrow(data_vendas), "observações\n")
  
  # ========================================
  # PREPARAÇÃO DOS DADOS
  # ========================================
  dados_preparados <- preparar_dados_vendas_completo(
    data = data_vendas,
    target_col = "vendas",
    seq_length = 30,
    prediction_horizon = 7,
    static_cols = NULL,
    categorical_cols = c("loja_id", "categoria_produto"),
    date_col = "data",
    include_calendar_features = TRUE,
    include_lag_features = TRUE,
    lag_periods = c(1, 7, 14),
    validation_split = 0.15,
    test_split = 0.15
  )
  
  # ========================================
  # CRIAÇÃO DO MODELO
  # ========================================
  modelo <- temporal_fusion_transformer_enhanced(
    input_size = ncol(dados_preparados$splits$train$X_past),
    output_size = 1,
    hidden_size = 64,
    num_heads = 4,
    num_layers = 2,
    dropout = 0.1,
    prediction_horizon = 7,
    num_static_vars = 0,
    num_time_varying_vars = ncol(dados_preparados$splits$train$X_past),
    categorical_cardinalities = dados_preparados$categorical_cardinalities,
    use_quantile_forecasting = TRUE,
    quantiles = c(0.1, 0.25, 0.5, 0.75, 0.9),
    use_locality_enhancement = TRUE,
    use_relative_attention = TRUE,
    encoder_length = 30
  )
  
  cat("Modelo TFT Enhanced criado!\n")
  total_params <- sum(sapply(modelo$parameters, function(p) prod(p$shape)))
  cat("Total de parâmetros:", format(total_params, big.mark = ","), "\n")
  
  # ========================================
  # TREINAMENTO
  # ========================================
  cat("\nIniciando treinamento...\n")
  
  resultado_treino <- treinar_tft_completo(
    modelo = modelo,
    dados_preparados = dados_preparados,
    epochs = 50,
    lr = 0.001,
    batch_size = 32,
    weight_decay = 1e-5,
    early_stopping_patience = 15
  )
  
  # ========================================
  # PREVISÕES E AVALIAÇÃO
  # ========================================
  cat("\nGenerating predictions...\n")
  
  # Previsões no conjunto de teste
  previsoes_teste <- prever_tft_completo(
    modelo = modelo,
    dados_preparados = dados_preparados,
    split = "test",
    return_quantiles = TRUE,
    return_attention = TRUE
  )
  
  # Calcular métricas
  metricas <- calcular_metricas_tft(
    predictions = previsoes_teste$predictions,
    targets = previsoes_teste$targets,
    quantiles = previsoes_teste$quantiles
  )
  
  # ========================================
  # RESULTADOS
  # ========================================
  cat("\n=== RESULTADOS FINAIS ===\n")
  cat("Épocas treinadas:", resultado_treino$final_epoch, "\n")
  cat("Melhor loss de validação:", sprintf("%.6f", resultado_treino$best_valid_loss), "\n")
  
  cat("\nMétricas no conjunto de teste:\n")
  cat("MAE:", sprintf("%.4f", metricas$MAE), "\n")
  cat("RMSE:", sprintf("%.4f", metricas$RMSE), "\n")
  cat("MAPE:", sprintf("%.2f%%", metricas$MAPE), "\n")
  cat("R²:", sprintf("%.4f", metricas$R2), "\n")
  
  if (!is.null(metricas$quantiles)) {
    cat("\nTaxas de cobertura dos quantis:\n")
    for (i in seq_along(metricas$quantiles)) {
      cat(sprintf("Q%.1f: %.3f (esperado: %.3f)\n", 
                 metricas$quantiles[i] * 100,
                 metricas$coverage_rates[i],
                 metricas$quantiles[i]))
    }
  }
  
  return(list(
    modelo = modelo,
    dados_preparados = dados_preparados,
    resultado_treino = resultado_treino,
    previsoes_teste = previsoes_teste,
    metricas = metricas
  ))
}

# Para executar o exemplo completo:
# resultado_completo <- exemplo_uso_tft_completo()
