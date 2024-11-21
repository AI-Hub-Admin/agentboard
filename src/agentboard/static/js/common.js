        function showBgImg(e) {
            var modal = document.getElementById('alert_box');
            var bgImg = document.getElementById('bgImg');

            modal.style.display = 'block';
            bgImg.src = e.src;
        }

        function bgImgClick() {
            var modal = document.getElementById('alert_box');
            modal.style.display = 'none';
        }


        function format_json(inputJson) {
            if (inputJson == null) {
                return ""
            }
            try {
                // indent with two space
                var prettyJson = JSON.stringify(JSON.parse(inputJson), null, 2);
                return prettyJson
            } catch (e) {
                // alert("Input Json is invalid" + inputJson);
                return inputJson
            }
        }

        function display_all_json() {
            var elements = document.querySelectorAll('.div_p_json');
            if (elements != null) {
                elements.forEach(function(element) {
                    var p_value = element.innerHTML;
                    element.innerHTML = format_json(p_value);
                });
            }
        }

        function copyCode(div_id) {
            // Get the code snippet text
            const codeSnippet = document.getElementById(div_id).innerText;

            // Use the clipboard API to copy text
            navigator.clipboard.writeText(codeSnippet).then(() => {
                alert("Code copied to clipboard!");
            }).catch(err => {
                console.error("Failed to copy text: ", err);
            });
        }

        function clickCompareLi(liIndex) {
            var lis = document.querySelectorAll('.compareTabLi');
            var items = document.querySelectorAll('.compareTabItem');
            if (lis == null || lis.length != items.length || items == null) {
                return;
            }
            //Change Switch TAB li
            for (var i = 0; i < lis.length; i++) {
                lis[i].className = 'compareTabLi';
            }
            lis[liIndex].className = 'compareTabLi tabLiClickColor';
            //Change Switch
            for (var i = 0; i < items.length; i++) {
                if (i == liIndex) {
                    items[i].style.display = 'block';
                } else {
                    items[i].style.display = 'none';
                }
            }
        }

