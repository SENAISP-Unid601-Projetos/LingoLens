# 🖐️ LingoLens – Tradutor de Libras em Tempo Real

O **LingoLens** é um tradutor de **Libras (alfabeto A-Z e números 0-9)** em tempo real. Ele captura gestos via câmera, processa os **landmarks da mão** e converte os sinais em texto, exibindo a palavra formada diretamente na tela.

---

## ▶️ Como Executar

Basta executar o arquivo **`start.bat`** no Windows.  

O script faz automaticamente:

- Verificação do Python e dependências necessárias.  
- Download/instalação de bibliotecas que não estiverem presentes.  
- Inicialização do aplicativo com interface gráfica para reconhecimento de gestos.  

---

## ⌨️ Atalhos no Programa

- **Q** → Sair  
- **C** → Limpar palavra  
- **N** → Alternar entre número/letra  
- **T** → Alternar modo Treinamento  
- **S** → Criar novo gesto (A-Z, 0-9)  
- **E** → Exportar gestos em JSON  
- **H** → Mostrar/ocultar ajuda  

---

## 🧪 Testes Unitários

O aplicativo inclui testes automáticos (`unittest`).  
Para executá-los manualmente (opcional):
```bash
python Main.py
```
Os testes verificarão funcionalidades básicas, como extração de landmarks e exportação de gestos.

---

## 🛠️ Logs e Mensagens de Erro

- Logs de execução e erros são salvos em **`gesture_recognizer.log`**.  
- Mensagens de erro também aparecem na interface do aplicativo, caso algum problema seja detectado (como ausência de mão na câmera).

---

link do Protipo do app 1° tela : https://www.figma.com/design/kLJRvJxt9TGZKUDxIblQcQ/PrototipoAppCelular?node-id=0-1&t=DbuCw4zNuMubBJTE-1

