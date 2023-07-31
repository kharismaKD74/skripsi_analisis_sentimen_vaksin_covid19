$.ajax({
  type: "GET",
  url: "http://127.0.0.1:5000/data/analisis",
  success: function (response) {
      console.log(
        response,
        typeof(response))
      
        var key = Object.keys(response)
        console.log(key)
        var val = Object.values(response)
        console.log(val)
        buatGrafik(key,val)
  }
})

function buatGrafik(key, val){
var grafik = echarts.init(
  document.getElementById("grafikPie")
)
var option = {
  title: {
    text: 'Analisis Sentimen Twitter',
    subtext: 'Vaksin Covid-19 di Indonesia',
    left: 'center'
  },
  tooltip: {
    trigger: 'item'
  },
  legend: {
    orient: 'vertical',
    left: 'left'
  },
  series: [
    {
      name: 'sentimen',
      type: 'pie',
      radius: '50%',
      data: [
        { value: val[0], name: key[0] },
        { value: val[1], name: key[1] },
        { value: val[2], name: key[2] }
      ],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }
  ]
};
grafik.setOption(option)
}
