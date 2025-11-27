# LingoLens – Tradutor de Datilologia Libras em Tempo Real

**LingoLens** reconhece em tempo real o alfabeto manual da Língua Brasileira de Sinais (datilologia A–Z), incluindo as letras com movimento: **H · J · K · X · Y · Z**.

---
## Funcionalidades

- Reconhecimento em tempo real das 26 letras  
- Suporte completo às letras dinâmicas com gravação por toggle (tecla T)  
- Treinamento interativo direto na câmera  
- Banco de dados persistente (SQLite)  
- Interface intuitiva com feedback visual  
- Arquitetura modular e profissional

---
## Atalhos do programa

- T: Treinar gesto 
- C: Limpar texto
- D: Excluir gesto
- L: Listar letras salvas 
- S: Salvar treino

---

## Como Executar

### 1-Para qualquer pessoa (recomendado)

## Download do Executável (Windows)

Link direto para o executável único (≈660 MB):  
[LingoLens-TCC-2025.exe – Google Drive](https://drive.google.com/file/d/1Njw8qRXg81pQZ-laUqlxQykUj3GDAOT1/view?usp=sharing)

Como usar:
1. Clique no link acima
2. Baixe o arquivo `LingoLens-TCC-2025.exe`
3. Dê dois cliques → abre a câmera e já funciona!

Funciona em qualquer Windows 10/11

### 2-Para quem quer ver o código(Clonar repositório)
→ Dê dois cliques em `start.bat` (cria tudo automaticamente)

### 3- Forma Manual
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python Main.py
```

**Qualquer sistema:**  

```bash
python main.py
```
## Autores

**LingoLens – Tradutor de Datilologia Libras em Tempo Real**  
Projeto desenvolvido pelo grupo:

- **Samuel Junior Casseta**  
- **Gabriel da Silva Rodrigues dos Santos**  
- **Abner Firmino Cerqueira**
- **Carlos Gabriel Evangelista Silva**  
- **Gustavo Ferreira Nunes**  


**Curso:** Técnico Desenvolvimento de Sistemas  
**Instituição:** Faculdade de Tecnologia e Escola SENAI Antonio Adolpho Lobbe  
**Ano:** 2025

---

## Licença

Este projeto está licenciado sob a **GNU General Public License v3.0** (GPLv3).  
Qualquer modificação ou distribuição deve manter a mesma licença e disponibilizar o código-fonte.

Veja o arquivo completo em [`LICENSE`](LICENSE)  
Mais informações: https://www.gnu.org/licenses/gpl-3.0.html
