executable_path := ./mangasplit.py
install_path := ~/.local/bin/mangasplit

install:
	cp $(executable_path) $(install_path) && chmod +x $(install_path)