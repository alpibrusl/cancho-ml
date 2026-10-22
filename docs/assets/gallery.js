// The gallery filter: shows the cards of one kind. Without this script every card is shown.
(function () {
  var bs = document.querySelectorAll(".filters button"), cs = document.querySelectorAll(".gcard");
  for (var i = 0; i < bs.length; i++) bs[i].addEventListener("click", function () {
    var f = this.getAttribute("data-f");
    for (var j = 0; j < bs.length; j++) bs[j].setAttribute("aria-pressed", bs[j] === this ? "true" : "false");
    for (var k = 0; k < cs.length; k++) cs[k].hidden = f !== "all" && cs[k].getAttribute("data-kind") !== f;
  });
})();
