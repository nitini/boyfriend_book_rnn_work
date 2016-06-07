window.onload = function() {
	var genders = document.getElementsByClassName("gender"),
		options = document.getElementsByClassName("option");

	function selectButton() {
		var classList = this.classList;

		if (classList.contains("gender")) {
			clearButtons(genders);
		} else {
			clearButtons(options);
		}

		if (this.style.backgroundColor === "transparent" || !this.style.backgroundColor) {
			this.style.backgroundColor = "#D3D3D3";
			this.style.opacity = "1";
		} else {
			this.style.backgroundColor = "transparent";
		}
	}

	function clearButtons(elementsArray) {
		for (var i = 0; i < elementsArray.length; i++) {
			elementsArray[i].style.backgroundColor = "transparent";
			elementsArray[i].style.opacity = ".7"
		}
	}

	for (var i = 0; i < genders.length; i++) {
		genders[i].addEventListener('click', selectButton, false);
	}

	for (var i = 0; i < options.length; i++) {
		options[i].addEventListener('click', selectButton, false);
	}
}
