###############################################################################
# PAKETE LADEN
###############################################################################
library(readxl)
library(lmerTest)    # für LMM
library(ggpubr)      # für Boxplots
library(glmmTMB)     # für Beta- (und andere GLMMs)
# -> Optional: library(DHARMa) für simulationsbasierte Residuenchecks

###############################################################################
# DATEN EINLESEN
###############################################################################
setwd("C:/Users/User/Desktop/Alina-SpikeSorting")
df <- read.csv("all_collected_scores.csv")

# Spalten, die ausgewertet werden sollen
columns <- c("F1", "Precision", "Recall", "Specificity", "FPR", "FNR")

###############################################################################
# HAUPTSCHLEIFE ÜBER ALLE Metriken
###############################################################################
epsilon <- 0.0001 # Offset, um exakte 0/1-Werte zu verschieben

for (col_name in columns) {
  
  #############################################################################
  # 1) KLEINER CHECK / SHIFT, UM EXAKTE 0 und 1 ZU VERMEIDEN
  #############################################################################
  shifted_col <- paste0(col_name, "_shifted")
  # In einer Kopie des Datensatzes anlegen (man kann es auch dauerhaft in df machen)
  df[[shifted_col]] <- pmin(pmax(df[[col_name]], epsilon), 1 - epsilon)
  
  #############################################################################
  # 2) MODELLFORMELN
  #############################################################################
  # z.B. ohne Random-Effekt (Fixed-Effects-Modell):
  formula_fixed <- as.formula(paste0(col_name, " ~ Model * FeatureSet"))
  
  # mit Random-Effekt (1 | Dataset) => gemischtes Modell
  formula_mixed <- as.formula(paste0(col_name, " ~ Model * FeatureSet + (1 | Dataset)"))
  
  # Beta-Formel nutzt shifted_col anstelle von col_name
  formula_beta <- as.formula(paste0(shifted_col, " ~ Model * FeatureSet + (1 | Dataset)"))
  
  #############################################################################
  # 3) VERSCHIEDENE MODELLE SCHÄTZEN
  #############################################################################
  
  # 3.1) Klassisches lineares Modell (Originalwerte)
  model_lm <- lm(formula_fixed, data = df)
  
  # 3.2) Lineares gemischtes Modell mit logit-Transformation
  # Achtung bei exakten 0/1 => offset
  logit_col <- paste0(col_name, "_logit")
  df[[logit_col]] <- log( (df[[col_name]] + epsilon) / (1 - df[[col_name]] + epsilon) )
  
  formula_lmer_logit <- as.formula(paste0(logit_col, " ~ Model * FeatureSet + (1 | Dataset)"))
  model_lmer_logit <- lmer(formula_lmer_logit, data = df)
  
  # 3.3) Beta-Gemischtes Modell (ohne Zero Inflation)
  model_beta <- glmmTMB(
    formula_beta,
    data = df,
    family = beta_family(link = "logit")
  )
  
  # 3.4) Optional: Zero-Inflation-Beta (nur wenn viele 0/1-Werte vorhanden)
  # Falls nötig, aktiviere diese Zeilen (ansonsten auskommentiert):
  # model_beta_zi <- glmmTMB(
  #   formula_beta,
  #   data = df,
  #   family = beta_family(link = "logit"),
  #   ziformula = ~1  # einfachstes Zero-Inflation-Modell mit (Intercept) 
  # )
  
  #############################################################################
  # 4) AIC-VERGLEICH
  #############################################################################
  # Modelle in Liste packen und AIC vergleichen
  # Achtung: LMM vs LM vs GLMM -> AIC ist nur bedingt 1:1 vergleichbar,
  #         aber als grober Hinweis durchaus nützlich.
  model_list <- list(
    LM             = model_lm,
    LMM_logit      = model_lmer_logit,
    Beta_mixed     = model_beta
    #Beta_zi_mixed = model_beta_zi
  )
  
  # AIC-Werte abrufen
  aic_vals <- sapply(model_list, AIC)
  
  # OPTIONAL: Auch BIC
  bic_vals <- sapply(model_list, BIC)
  
  # SUMMARY-Ausgaben sammeln
  sink(paste0("Summary_Models_", col_name, ".txt"))
  cat("===== Modellvergleiche für", col_name, "=====\n")
  
  cat("\n*** AIC-Werte ***\n")
  print(aic_vals)
  
  cat("\n*** BIC-Werte ***\n")
  print(bic_vals)
  
  cat("\n\n=== LM (ohne Random) ===\n")
  print(summary(model_lm))
  
  cat("\n\n=== LMM mit logit-Transform ===\n")
  print(summary(model_lmer_logit))
  
  cat("\n\n=== Beta-Mixed ===\n")
  print(summary(model_beta))
  
  # Auskommentiert, falls Du Zero Inflation aktivierst:
  # cat("\n\n=== Beta-ZI-Mixed ===\n")
  # print(summary(model_beta_zi))
  
  sink()  # Schliesst die Datei
  
  #############################################################################
  # 5) DIAGNOSEPLOTS (nur als erste Anhaltspunkte)
  #    Du kannst Dir pro Modell Residuenplot, QQ-Plot etc. anschauen, 
  #    um ein Gefühl für die Modellgüte zu bekommen.
  #############################################################################
  # Beispiel: QQ-Plots für Residuen 
  # (Beachte: bei Beta-Regression können klassische QQ-Plots irreführend sein!)
  
  # LM-Modell
  jpeg(paste0("QQPlot_LM_", col_name, ".jpeg"))
  qqnorm(residuals(model_lm), main = paste("QQ Plot LM", col_name))
  qqline(residuals(model_lm))
  dev.off()
  
  # LMM-Logit-Modell
  jpeg(paste0("QQPlot_LMM_Logit_", col_name, ".jpeg"))
  qqnorm(residuals(model_lmer_logit), main = paste("QQ Plot LMM Logit", col_name))
  qqline(residuals(model_lmer_logit))
  dev.off()
  
  # Beta-Mixed-Modell
  jpeg(paste0("QQPlot_Beta_", col_name, ".jpeg"))
  qqnorm(residuals(model_beta), main = paste("QQ Plot Beta Mixed", col_name))
  qqline(residuals(model_beta))
  dev.off()
  
  # Boxplot (Originalwerte, um die Verteilung nach Model/FeatureSet anzusehen)
  jpeg(paste0("BoxPlot_", col_name, ".jpeg"))
  bxp <- ggboxplot(
    data = df, 
    x = "Model", 
    y = col_name,
    palette = "jco", 
    facet.by = "FeatureSet",
    short.panel.labs = FALSE
  )
  print(bxp)
  dev.off()
  
  # Histogramm der Originalwerte
  jpeg(paste0("Histogram_", col_name, ".jpeg"))
  hist(df[[col_name]], main = paste("Histogram of", col_name), xlab = col_name)
  dev.off()
  
  # Beispielhafter Residuen-vs-Fitted-Plot für Beta-Mixed
  jpeg(paste0("Residuals_vs_Fitted_Beta_", col_name, ".jpeg"))
  plot(fitted(model_beta), residuals(model_beta),
       main = paste("Residuen vs. Fitted (Beta)", col_name),
       xlab = "Fitted Values", ylab = "Residuals")
  abline(h=0, col="red")
  dev.off()
  
  jpeg(paste0("Residuals_vs_Fitted_Beta2_", col_name, ".jpeg"))
  resid_pearson <- residuals(model_beta, type = "pearson")
  fit <- predict(model_beta, type="response")  # vorhergesagte Werte auf der Originalskala
  plot(fit, resid_pearson,
       main = "Pearson Residuals vs Fitted",
       xlab = "Fitted (Predicted) Values",
       ylab = "Pearson Residuals")
  abline(h = 0, col = "red")
  dev.off()
}

###############################################################################
# TIPP: DHARMa für GLMM-Residuals
###############################################################################
# Falls Du die Modellresiduen für glmmTMB modelle wie Beta oder Beta-ZI
# genauer checken willst, kannst Du "DHARMa" nutzen:
#
# library(DHARMa)
# simulationOutput <- simulateResiduals(fittedModel = model_beta)
# plot(simulationOutput)
#
# Dann erhältst Du einheitliche Residualdiagnostik (QQ, Dispersion, etc.).
###############################################################################