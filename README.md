# Proteção de Crianças

Aplicação local em **Python + Streamlit + OpenCV** para detectar rostos em fotografias, escolher quais devem ser protegidos, aplicar desfoque e revisar o resultado antes da exportação.

## Privacidade

- A aplicação não precisa enviar a fotografia para APIs externas.
- O processamento ocorre na máquina onde o Streamlit está sendo executado.
- As imagens ficam na sessão da aplicação e não são gravadas permanentemente pelo código do projeto.
- Sempre revise o resultado: detectores automáticos podem falhar ou não identificar todos os rostos.

## Funcionalidades

- Upload JPG, JPEG e PNG.
- Detecção automática de rostos com Haar Cascade do OpenCV.
- Numeração e marcação visual dos rostos encontrados.
- Seleção individual, selecionar todos e desmarcar todos.
- Inclusão de área manual quando a detecção automática não for suficiente.
- Ajuste de intensidade do blur e margem de proteção.
- Comparação entre imagem original e protegida.
- Etapa de confirmação final.
- Download em PNG.
- Navegação reversível para corrigir seleções.

## Estrutura

```text
PROTECAO-CRIANCAS/
├── assets/
├── src/
│   ├── __init__.py
│   ├── detector.py
│   └── processor.py
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Como executar no Windows

### 1. Instalar Python

Recomendado: Python 3.11 ou 3.12, 64 bits.
Durante a instalação, marque **Add Python to PATH**.

### 2. Abrir o Prompt de Comando na pasta do projeto

Exemplo:

```bat
cd C:\PROTECAO-CRIANCAS
```

### 3. Criar ambiente virtual

```bat
python -m venv .venv
```

### 4. Ativar o ambiente

```bat
.venv\Scripts\activate
```

### 5. Instalar dependências

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Executar

```bat
streamlit run app.py
```

O navegador normalmente abrirá automaticamente. Se não abrir, use o endereço exibido no Prompt de Comando, geralmente `http://localhost:8501`.

## Como transformar em EXE

Streamlit é uma aplicação web local; por isso um `.exe` precisa iniciar o servidor Streamlit e abrir o navegador. O caminho mais confiável é usar **PyInstaller** com um pequeno launcher.

### 1. Com o ambiente virtual ativado, instale o PyInstaller

```bat
pip install pyinstaller
```

### 2. Na pasta imediatamente acima de `PROTECAO-CRIANCAS`, crie temporariamente `launcher.py`

```python
import os
import sys
from streamlit.web import cli as stcli

base = os.path.dirname(os.path.abspath(__file__))
app = os.path.join(base, "PROTECAO-CRIANCAS", "app.py")
sys.argv = ["streamlit", "run", app, "--server.headless=false", "--browser.gatherUsageStats=false"]
sys.exit(stcli.main())
```

### 3. Gere o executável

No Prompt de Comando, na pasta que contém `launcher.py` e `PROTECAO-CRIANCAS`:

```bat
pyinstaller --noconfirm --clean --name ProtecaoCriancas --collect-all streamlit --collect-all cv2 --collect-all PIL --add-data "PROTECAO-CRIANCAS;PROTECAO-CRIANCAS" launcher.py
```

O executável ficará em:

```text
dist\ProtecaoCriancas\ProtecaoCriancas.exe
```

> Para Streamlit, a modalidade `--onedir` é recomendada. `--onefile` pode ficar muito grande e demorar mais para abrir.

## Fluxo de uso

1. Clique em **Começar**.
2. Carregue a fotografia.
3. Aguarde a detecção dos rostos.
4. Confira as caixas encontradas.
5. Selecione os rostos/áreas que devem ser protegidos.
6. Caso necessário, adicione uma área manual.
7. Clique em **Aplicar proteção**.
8. Compare original e imagem protegida.
9. Volte para corrigir se necessário.
10. Confirme a última revisão.
11. Baixe a imagem protegida.

## Observação técnica

O detector Haar Cascade é propositalmente local e simples. Ele é adequado para o protótipo e para o objetivo acadêmico, mas pode falhar em rostos de perfil, parcialmente cobertos, muito pequenos ou em iluminação difícil. A revisão humana e a correção manual continuam essenciais.


## Inicialização direta (sem CMD)

Para abrir o sistema sem digitar comandos:

1. extraia a pasta do projeto;
2. dê dois cliques em **INICIAR.bat**;
3. se preferir, use **INICIAR.vbs** para iniciar pelo atalho visual do Windows;
4. o sistema abrirá no navegador em `http://localhost:8501`.

Na primeira execução, o arquivo `INICIAR.bat` cria o ambiente virtual e instala as dependências automaticamente.


## Blur manual com mouse

Logo após a etapa **Detecção**, o botão **Usar borrão manual** abre o editor no topo da etapa **Proteção**. Ele fica disponível sempre, com ou sem rostos detectados, e serve para rostos não reconhecidos, cortados na borda da foto, de perfil ou parcialmente cobertos. Pelo botão **Confirmar detecção**, o mesmo editor aparece abaixo da lista de rostos.

1. clique e arraste sobre qualquer região da imagem para criar um quadrante manual;
2. use o botão **Editar** da barra do próprio editor para selecionar uma caixa;
3. arraste a caixa para mudar a posição;
4. arraste as alças para aumentar ou diminuir;
5. use **Apagar** para remover uma área específica ou **Limpar todas**;
6. clique em **Aplicar proteção e revisar** para gerar a imagem final;
7. após a revisão e confirmação, use **Baixar imagem**.

A versão atual usa `streamlit-drawable-canvas` 0.13.x. Nessa versão a edição não usa mais `drawing_mode="transform"`; mover e redimensionar são feitos pelo botão de edição da barra do canvas.

Se, no modo de edição, um canto da caixa for arrastado e ela deixar de ser um retângulo, o desfoque cobre o retângulo que envolve a forma.
