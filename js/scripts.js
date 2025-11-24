/*!
* Start Bootstrap - Agency v7.0.12 (https://startbootstrap.com/theme/agency)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-agency/blob/master/LICENSE)
*/
// Scripts
//ts-ignore
window.addEventListener('DOMContentLoaded', event => {

    // ----------------------------------------
    // NAVBAR / BOOTSTRAP ORIGINAL
    // ----------------------------------------

    var navbarShrink = function () {
        const navbarCollapsible = document.body.querySelector('#mainNav');
        if (!navbarCollapsible) return;

        if (window.scrollY === 0) {
            navbarCollapsible.classList.remove('navbar-shrink')
        } else {
            navbarCollapsible.classList.add('navbar-shrink')
        }
    };

    navbarShrink();
    document.addEventListener('scroll', navbarShrink);

    const mainNav = document.body.querySelector('#mainNav');
    if (mainNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#mainNav',
            rootMargin: '0px 0px -40%',
        });
    }

    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

    // ----------------------------------------
    // FORMULÁRIO EMAILJS
    // ----------------------------------------
    initContactForm();
});


// =====================================================================
// FORMULÁRIO DE CONTATO — VERSÃO 100% SEM DUPLICAÇÃO
// =====================================================================

function initContactForm() {

    const form = document.getElementById("contactForm");
    if (!form) return;

    const successMsg = document.getElementById("submitSuccessMessage");
    const errorMsg = document.getElementById("submitErrorMessage");
    const submitBtn = document.getElementById("submitButton");

    // Inicializa EmailJS
    emailjs.init("jyrzAnfUExQYePIcq");

    // Remove listeners duplicados
    form.replaceWith(form.cloneNode(true));
    const newForm = document.getElementById("contactForm");

    newForm.addEventListener("submit", function (e) {
        e.preventDefault();

        if (!newForm.checkValidity()) {
            newForm.classList.add("was-validated");
            return;
        }

        submitBtn.disabled = true;
        submitBtn.innerText = "Enviando...";

        const params = {
            title: document.getElementById("name").value,
            email: document.getElementById("email").value,
            phone: document.getElementById("phone").value,
            message: document.getElementById("message").value,
        };

        console.log("Enviando para EmailJS:", params);

        emailjs.send("service_qdapsba", "template_gqb7r7t", params)
            .then(() => {
                newForm.reset();
                newForm.classList.remove("was-validated");

                successMsg.classList.remove("d-none");
                errorMsg.classList.add("d-none");

                submitBtn.disabled = false;
                submitBtn.innerText = "Enviar";
            })
            .catch((err) => {
                console.error("Erro EmailJS:", err);

                errorMsg.classList.remove("d-none");
                successMsg.classList.add("d-none");

                submitBtn.disabled = false;
                submitBtn.innerText = "Enviar";
            });
    });
}

