#!/usr/bin/awk -f
# Script awk para selecionar:
# 1. A segunda linha do arquivo
# 2. A primeira linha onde aparece "Restoring intermediary model from checkpoint..."
# E exibir apenas a segunda coluna (horário)
# Processa múltiplos arquivos de um diretório

BEGINFILE {
    found = 0
}

FNR == 2 { 
    printf "%s train_ini: '%s', ", $7, $2
}

/Checkpointing latest model at step 10000/ && !found { 
    printf "train_fim: '%s' \n", $2
    found = 0
}