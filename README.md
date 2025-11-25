# LingoLens – Tradutor de Datilologia Libras em Tempo Real

**LingoLens** é um sistema de reconhecimento em tempo real da **datilologia da Língua Brasileira de Sinais (Libras)** — o alfabeto manual de A a Z.

O projeto identifica com alta precisão tanto as letras estáticas quanto as seis letras que possuem movimento obrigatório na Libras: **H · J · K · X · Y · Z**.

---

## Funcionalidades Principais

- Reconhecimento em tempo real das 26 letras do alfabeto datilológico  
- Suporte completo às letras dinâmicas (H, J, K, X, Y, Z) com coleta via toggle (tecla T)  
- Treinamento interativo diretamente na câmera  
- Persistência automática dos gestos (SQLite + pickle)  
- Interface intuitiva com feedback visual em tempo real  
- Arquitetura limpa e modular (pronta para expansão futura)

---

## Como Executar

### Windows (recomendado)
Clique duas vezes no arquivo **`start.bat`**  
→ Ele instala todas as dependências automaticamente e inicia o programa.

### Qualquer sistema
```bash
python main.py