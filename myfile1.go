package main

//#include "mconf.h"
//#cgo LDFLAGS: -L . ./randomkit.so
//#cgo LDFLAGS: -L . ./distributions.so
//#cgo LDFLAGS: -L . ./psi.so
//#cgo LDFLAGS: -L . ./gamma.so
//#include "randomkit.h"
//#include "distributions.h"
import "C"
import (
	"fmt"
)
func main() {
	// st := &C.struct_rk_state_{pos: 100}
	// lng := C.rk_random(st)

	// hsh := C.rk_long(st)

	// fmt.Println(lng)
	// fmt.Println(hsh)

	st := &C.struct_rk_state_{}
	err := C.rk_randomseed(st)
	fmt.Println("err", err)
	//fmt.Println(st.pos)
	//fmt.Println(st.key)

	//C.rk_seed(1, st)
	//fmt.Println(st.pos)
	//fmt.Println(st.key)

	fmt.Println("gamma value")
	result := C.rk_gamma(st, 100., 1./100.)
	fmt.Println(result)

	fmt.Println("psi value")
	//var x float64 = 2.0
	result1 := C.psi(result)
	fmt.Println(result1)

	/*gamma value
	1.1677842454242098
	psi value
	-0.3306549565913235*/

	fmt.Println("lgamma value")
	//var x float64 = 2.0
	result2 := C.gamma(67.)
	fmt.Println(result2)
}
