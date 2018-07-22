library('ggplot2')
library('forecast')
library('tseries')
library('keras')
library('recipes')
library('timetk')
library('tidyquant')
library('tibbletime')
library('glue')
library('yardstick')

library('tidyverse')

install_keras()

bean <- read.csv("bean.csv", stringsAsFactors = FALSE) %>% 
  tk_tbl() %>% 
  mutate(index=as_date(date)) %>%
  as_tbl_time(index=index) %>%
  as.double(price)

bean <- subset(bean, select = -c(date))


bean$price <- as.double(sub(",", "", bean$price))

ggplot(bean, aes(index, price)) + geom_line() + scale_x_date('Time')  + ylab("Price") +
  xlab("")

bean$ma <- ma(bean$price, order=7)
bean$ma30 <- ma(bean$price, order=30)


#ma process 확인
ggplot() +
  geom_line(data = bean, aes(x = date, y = price, colour = "Price")) +
  geom_line(data = bean, aes(x = date, y = ma,   colour = "Weekly Moving Average"))  +
  geom_line(data = bean, aes(x = date, y = ma30, colour = "Monthly Moving Average"))  +
  ylab('Price')


#성분 분해
ma <- ts(na.omit(bean$ma), frequency = 30)
decomp <- stl(ma, s.window = "periodic")
deseasonal_price <- seasadj(decomp)
plot(decomp)

adf.test(ma, alternative = "stationary")

Acf(ma, main='')
Pacf(ma, main='')

sprice <- diff(bean$price, differences = 2)
dprice <- diff(deseasonal_price, differences = 2)
plot(dprice)

adf.test(dprice, alternative = "stationary")


fit <- auto.arima(sprice, seasonal = FALSE)
tsdisplay(residuals(fit), lag.max = 45, main='')

fcast <- forecast(fit, h=14)
plot(fcast)

seas_fit <- auto.arima(dprice, seasonal = TRUE)
seas_fcast <- forecast(seas_fit, h=14)
plot(seas_fcast)


#LSTM
train <- bean[1:2200, ] 
test <- bean[2201:2300, ]

df <- bind_rows(
  train %>% add_column(key="train"),
  test %>% add_column(key="test")
) %>% as_tbl_time(index=index)


rec_obj <- recipe(price ~ ., df) %>%
  step_sqrt(price) %>%
  step_center(price) %>%
  step_scale(price) %>%
  prep()

df_processed_tbl <- bake(rec_obj, df)

center_history <- rec_obj$steps[[2]]$means["price"]
scale_history  <- rec_obj$steps[[3]]$sds["price"]

lag_setting  <- 100
batch_size   <- 25
train_length <- 2300
tsteps       <- 1
epochs       <- 500


lag_train_tbl <- df_processed_tbl %>%
  mutate(value_lag = lag(price, n = lag_setting)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "train") %>%
  tail(train_length)

x_train_vec <- lag_train_tbl$value_lag
x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))

y_train_vec <- lag_train_tbl$price
y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))

# Testing Set
lag_test_tbl <- df_processed_tbl %>%
  mutate(
    value_lag = lag(price, n = lag_setting)
  ) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "test")

x_test_vec <- lag_test_tbl$value_lag
x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))

y_test_vec <- lag_test_tbl$price
y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), 1))



model <- keras_model_sequential()

model %>%
  layer_lstm(units            = 50, 
             input_shape      = c(tsteps, 1), 
             batch_size       = batch_size,
             return_sequences = TRUE, 
             stateful         = TRUE) %>%
  layer_dropout(rate = 0.4) %>%
  layer_lstm(units            = 50, 
             return_sequences = FALSE, 
             stateful         = TRUE) %>% 
  layer_dense(units = 1)

model %>% 
  compile(loss = 'mae', optimizer = 'adam')

for (i in 1:epochs) {
  model %>% fit(x          = x_train_arr, 
                y          = y_train_arr, 
                batch_size = batch_size,
                epochs     = 1, 
                verbose    = 1, 
                shuffle    = FALSE)
  
  model %>% reset_states()
  cat("Epoch: ", i)
  
}

# Make Predictions
pred_out <- model %>% 
  predict(x_test_arr, batch_size = batch_size) %>%
  .[,1] 

# Retransform values
pred_tbl <- tibble(
  index   = lag_test_tbl$index,
  price   = (pred_out * scale_history + center_history)^2
) 

# Combine actual data with predictions
tbl_1 <- train %>%
  add_column(key = "actual")

tbl_2 <- test %>%
  add_column(key = "actual")

tbl_3 <- pred_tbl %>%
  add_column(key = "predict")

# Create time_bind_rows() to solve dplyr issue
time_bind_rows <- function(data_1, data_2, index) {
  index_expr <- enquo(index)
  bind_rows(data_1, data_2) %>%
    as_tbl_time(index = !! index_expr)
}

ret <- list(tbl_1, tbl_2, tbl_3) %>%
  reduce(time_bind_rows, index = index) %>%
  arrange(key, index) %>%
  mutate(key = as_factor(key))

calc_rmse <- function(prediction_tbl) {
  
  rmse_calculation <- function(data) {
    data %>%
      spread(key = key, value = price) %>%
      select(-index) %>%
      filter(!is.na(predict)) %>%
      rename(
        truth    = actual,
        estimate = predict
      ) %>%
      rmse(truth, estimate)
  }
  
  safe_rmse <- possibly(rmse_calculation, otherwise = NA)
  
  safe_rmse(prediction_tbl)
}



plot_prediction <- function(data, id, alpha = 1, size = 2, base_size = 14) {
  
  rmse_val <- calc_rmse(data)
  
  g <- data %>%
    ggplot(aes(index, price, color = key)) +
    geom_point(alpha = alpha, size = size) + 
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    theme(legend.position = "none") +
    labs(
      title = glue("RMSE: {round(rmse_val, digits = 1)}"),
      x = "", y = ""
    )
  
  return(g)
}
ret %>% 
  plot_prediction(alpha = 1) +
  theme(legend.position = "bottom")

  